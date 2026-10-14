"""
buscar_masivo.py
================
Analiza MUCHAS estrellas de TESS, una tras otra, y separa lo que ya se conoce
de lo que podría ser nuevo.

- Guarda el avance después de CADA estrella (progreso.jsonl). Si se corta la
  luz, se apaga el PC o lo detienes con Ctrl+C, al volver a ejecutar el mismo
  comando retoma donde quedó, sin repetir nada.
- Cada estrella corre en un proceso aparte con tiempo máximo: si una se cuelga,
  se descarta y sigue con la siguiente.
- Cruza cada señal con 3 catálogos: candidatos TESS (TOI), candidatos de la
  comunidad (CTOI) y planetas confirmados. Marca cada una como:
      CONOCIDO           -> ya está en un catálogo con ese período (o múltiplo)
      PERÍODO NUEVO      -> la estrella tiene objetos conocidos, pero no con ese período
      NUEVO              -> la estrella no tiene nada en ningún catálogo
- Al final (y cada vez que se ejecuta) escribe resultados.csv y resumen.md con
  los candidatos NUEVOS ordenados por la probabilidad del modelo de IA.

Ejemplos:
    # 50 estrellas al azar del sector 10, entre magnitud 9 y 12, que NO tengan TOI
    python buscar_masivo.py --sector 10 --max 50 --tmag 9 12 --sin-conocidos

    # Una lista propia (un TIC o nombre por línea)
    python buscar_masivo.py --lista mis_estrellas.txt

    # Solo regenerar el resumen con lo ya analizado
    python buscar_masivo.py --salida resultados_masivo --solo-resumen

Importante: "NUEVO" significa "no está en estos catálogos", NO "planeta
descubierto". Hay que revisar cada uno a mano (gráfico + README, sección límites).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import random
import shutil
import sys
import time
import traceback
import urllib.request
from datetime import datetime

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

URL = {
    "toi": "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv",
    "ctoi": "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv",
    "confirmados": ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,"
                    "tic_id,pl_orbper+from+pscomppars+where+tic_id+is+not+null&format=csv"),
    "objetivos": "https://tess.mit.edu/wp-content/uploads/all_targets_S{:03d}_v1.csv",
}
ARCH = {"toi": "toi_exofop.csv", "ctoi": "ctoi_exofop.csv", "confirmados": "confirmados.csv"}


def log(msg, salida=None):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    if salida:
        with open(os.path.join(salida, "log.txt"), "a", encoding="utf-8") as fh:
            fh.write(linea + "\n")


def descargar_archivo(url, destino, intentos=3):
    for i in range(intentos):
        try:
            tmp = destino + ".tmp"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, destino)            # nunca deja un archivo a medias
            return True
        except Exception as e:                  # red caída, servidor lento...
            log(f"  descarga fallida ({i+1}/{intentos}): {e}")
            time.sleep(5 * (i + 1))
    return False


# ---------------------------------------------------------------------------
# Catálogos
# ---------------------------------------------------------------------------
def cargar_catalogos(carpeta, actualizar=False):
    """Devuelve {tic: [(fuente, nombre, periodo, disposicion), ...]}."""
    import pandas as pd
    os.makedirs(carpeta, exist_ok=True)
    for clave, nombre in ARCH.items():
        ruta = os.path.join(carpeta, nombre)
        if actualizar or not os.path.exists(ruta):
            log(f"Descargando catálogo {clave}...")
            descargar_archivo(URL[clave], ruta)
    conocidos = {}

    def agregar(tic, fuente, nombre, periodo, disp):
        try:
            conocidos.setdefault(int(tic), []).append((fuente, str(nombre), float(periodo), str(disp)))
        except (TypeError, ValueError):
            pass

    p = os.path.join(carpeta, ARCH["toi"])
    if os.path.exists(p):
        for _, f in pd.read_csv(p, low_memory=False).iterrows():
            agregar(f["TIC ID"], "TOI", f"TOI {f['TOI']}", f["Period (days)"], f["TFOPWG Disposition"])
    p = os.path.join(carpeta, ARCH["ctoi"])
    if os.path.exists(p):
        for _, f in pd.read_csv(p, low_memory=False).iterrows():
            agregar(f["TIC ID"], "CTOI", f"CTOI {f['CTOI']}", f["Period (days)"], f["User Disposition"])
    p = os.path.join(carpeta, ARCH["confirmados"])
    if os.path.exists(p):
        for _, f in pd.read_csv(p).iterrows():
            tic = str(f["tic_id"]).replace("TIC", "").strip()
            agregar(tic, "confirmado", f["pl_name"], f["pl_orbper"], "confirmado")
    return conocidos


def cruzar(tic, periodo, conocidos):
    """Compara una señal con lo que ya se conoce de esa estrella."""
    objetos = conocidos.get(int(tic), [])
    if not objetos:
        return "NUEVO", ""
    for fuente, nombre, P, disp in objetos:
        if not (P == P and P > 0):              # período desconocido (NaN) en el catálogo
            continue
        for k in (1, 2, 3, 1 / 2, 1 / 3):
            if abs(periodo / (P * k) - 1) < 0.005:
                extra = "" if k == 1 else f" (alias x{k:.2g})"
                return "CONOCIDO", f"{nombre} [{fuente}, {disp}]{extra}"
    lista = "; ".join(f"{n} P={P:.3f}" for _, n, P, _ in objetos[:4])
    return "PERÍODO NUEVO", f"la estrella tiene: {lista}"


# ---------------------------------------------------------------------------
# Lista de estrellas
# ---------------------------------------------------------------------------
def lista_desde_sector(sector, salida, tmag, maximo, semilla, excluir):
    ruta = os.path.join(salida, f"objetivos_S{sector:03d}.csv")
    if not os.path.exists(ruta):
        log(f"Descargando lista de estrellas observadas por TESS en el sector {sector}...", salida)
        if not descargar_archivo(URL["objetivos"].format(sector), ruta):
            raise SystemExit("No pude descargar la lista del sector. Revisa tu conexión.")
    tics = []
    with open(ruta, encoding="utf-8") as fh:
        filas = (l for l in fh if not l.startswith("#"))
        for f in csv.DictReader(filas):
            try:
                t, m = int(f["TICID"]), float(f["Tmag"])
            except (KeyError, ValueError):
                continue
            if tmag and not (tmag[0] <= m <= tmag[1]):
                continue
            if t in excluir:
                continue
            tics.append(t)
    random.Random(semilla).shuffle(tics)       # misma semilla -> misma lista (reanudable)
    return [f"TIC {t}" for t in (tics[:maximo] if maximo else tics)]


def lista_desde_archivo(ruta):
    with open(ruta, encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip() and not l.startswith("#")]


def tic_de(objetivo):
    s = objetivo.upper().replace("TIC", "").strip()
    return int(s) if s.isdigit() else None


# ---------------------------------------------------------------------------
# Análisis de UNA estrella (corre en un proceso aparte)
# ---------------------------------------------------------------------------
def borrar_descargas(tic):
    import lightkurve as lk
    base = os.path.join(lk.config.get_cache_dir(), "mastDownload")
    marca = f"{tic:016d}"
    for raiz, dirs, _ in os.walk(base):
        for d in dirs:
            if marca in d:
                shutil.rmtree(os.path.join(raiz, d), ignore_errors=True)


def trabajador(objetivo, opciones, cola):
    import io
    import contextlib
    import warnings
    warnings.filterwarnings("ignore")
    t_ini = time.time()
    res = {"objetivo": objetivo, "estado": "error", "senales": []}
    try:
        import buscador_transitos as bt
        with contextlib.redirect_stdout(io.StringIO()):          # silencio
            lc, autor = bt.descargar(objetivo, opciones["sectores"], None)
        est = bt.parametros_estrella(lc)
        if opciones.get("invertir"):            # prueba nula: las caídas pasan a ser subidas
            lc = lc.remove_nans().normalize()
            lc.flux = 2 - lc.flux
        res["tic"] = tic_de(objetivo) or int(lc.meta.get("TICID") or 0)
        res.update(autor=autor, radio_estrella=None if est.supuesta else est.radio_sol,
                   teff=est.teff if est.teff == est.teff else None)
        senales, plano, pg = bt.analizar(lc, est, max_senales=opciones["max_senales"], verbose=False)
        if opciones.get("clasificador"):
            bt.aplicar_clasificador(senales, est, opciones["clasificador"])
        hay_candidato = any(s.veredicto == "CANDIDATO PLANETARIO" for s in senales)
        if hay_candidato:
            os.makedirs(opciones["graficos"], exist_ok=True)
            with contextlib.redirect_stdout(io.StringIO()):
                bt.graficar(plano, senales, pg, objetivo,
                            os.path.join(opciones["graficos"], f"TIC_{res['tic']}.png"))
        for s in senales:
            res["senales"].append(dict(
                numero=s.numero, periodo=s.periodo, t0=s.t0, duracion_h=s.duracion_h,
                profundidad_ppm=s.profundidad_central_ppm, radio_tierras=s.radio_tierras,
                sde=s.sde, snr=s.snr, n_transitos=s.n_transitos, veredicto=s.veredicto,
                prob_ia=s.prob_planeta if s.prob_planeta == s.prob_planeta else None,
                motivos=s.motivos,
                pruebas={k: (v if isinstance(v, str) else float(v)) for k, v in s.pruebas.items()}))
        res["estado"] = "ok"
    except SystemExit as e:                     # p. ej. "no hay datos de TESS"
        res.update(estado="sin_datos", error=str(e))
    except BaseException as e:
        res.update(error=f"{type(e).__name__}: {e}", detalle=traceback.format_exc()[-1500:])
    finally:
        res["segundos"] = round(time.time() - t_ini, 1)
        if opciones.get("borrar_datos") and res.get("tic"):
            try:
                borrar_descargas(res["tic"])
            except Exception:
                pass
        cola.put(res)


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
def leer_progreso(salida):
    ruta = os.path.join(salida, "progreso.jsonl")
    hechos = {}
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as fh:
            for linea in fh:
                try:
                    r = json.loads(linea)
                    hechos[r["objetivo"]] = r
                except json.JSONDecodeError:    # última línea cortada por un apagón
                    continue
    return hechos


def guardar_progreso(salida, res):
    with open(os.path.join(salida, "progreso.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())                   # que quede en disco aunque se corte la luz


def escribir_resumen(salida, conocidos):
    hechos = leer_progreso(salida)
    filas = []
    for r in hechos.values():
        for s in r.get("senales", []):
            if s["veredicto"] in ("SIN DETECCIÓN",):
                continue
            estado, detalle = cruzar(r.get("tic") or 0, s["periodo"], conocidos)
            filas.append(dict(objetivo=r["objetivo"], tic=r.get("tic"), senal=s["numero"],
                              veredicto=s["veredicto"], catalogo=estado, detalle_catalogo=detalle,
                              periodo_d=round(s["periodo"], 5), duracion_h=round(s["duracion_h"], 2),
                              profundidad_ppm=round(s["profundidad_ppm"]),
                              radio_tierras=round(s["radio_tierras"], 2), sde=round(s["sde"], 1),
                              snr=round(s["snr"], 1), prob_ia=s["prob_ia"],
                              motivos=" | ".join(s["motivos"])))
    campos = ["objetivo", "tic", "senal", "veredicto", "catalogo", "detalle_catalogo", "periodo_d",
              "duracion_h", "profundidad_ppm", "radio_tierras", "sde", "snr", "prob_ia", "motivos"]
    with open(os.path.join(salida, "resultados.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)

    estados = {}
    for r in hechos.values():
        estados[r["estado"]] = estados.get(r["estado"], 0) + 1
    cand = [f for f in filas if f["veredicto"] == "CANDIDATO PLANETARIO"]
    nuevos = sorted([f for f in cand if f["catalogo"] != "CONOCIDO"],
                    key=lambda f: -(f["prob_ia"] if f["prob_ia"] is not None else -1))
    conocidos_ok = [f for f in cand if f["catalogo"] == "CONOCIDO"]
    with open(os.path.join(salida, "resumen.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Búsqueda masiva — {datetime.now():%Y-%m-%d %H:%M}\n\n")
        fh.write(f"Estrellas procesadas: {len(hechos)} · " +
                 " · ".join(f"{k}: {v}" for k, v in sorted(estados.items())) + "\n\n")
        fh.write(f"Candidatos que pasan todas las pruebas: {len(cand)} "
                 f"({len(conocidos_ok)} ya conocidos, {len(nuevos)} sin registro)\n\n")
        fh.write("## Candidatos SIN registro en catálogos (revisar a mano)\n\n")
        if not nuevos:
            fh.write("Ninguno por ahora.\n\n")
        else:
            fh.write("| Estrella | Período (d) | Radio (R⊕) | Prob. IA | SNR | Catálogo | Advertencias |\n"
                     "|---|---|---|---|---|---|---|\n")
            for f in nuevos:
                adv = "; ".join(m for m in f["motivos"].split(" | ") if m.startswith("ADVERTENCIA"))
                p = f"{f['prob_ia']:.0%}" if f["prob_ia"] is not None else "—"
                fh.write(f"| {f['objetivo']} | {f['periodo_d']} | {f['radio_tierras']} | {p} | "
                         f"{f['snr']} | {f['catalogo']} | {adv[:160]} |\n")
        fh.write("\n## Candidatos que ya estaban en catálogos (sirven para ver que funciona)\n\n")
        for f in conocidos_ok:
            fh.write(f"- {f['objetivo']}: P={f['periodo_d']} d → {f['detalle_catalogo']}\n")
        fh.write("\n## Cómo revisar un candidato sin registro\n\n"
                 "1. Abre su gráfico en `graficos/`: ¿se ven tránsitos claros, iguales entre pares e impares, "
                 "sin nada en la mitad de la órbita?\n"
                 "2. Vuelve a analizarlo con más sectores: `python buscador_transitos.py --estrella \"TIC ...\" "
                 "--sectores 0 --clasificador ml/clasificador_toi.joblib`. Si es real, debe reaparecer.\n"
                 "3. Búscalo en https://exofop.ipac.caltech.edu/tess/ (puede haberse agregado hace poco).\n"
                 "4. Recién ahí se considera reportarlo como CTOI en ExoFOP (requiere cuenta).\n")
    return len(hechos), len(cand), len(nuevos)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Búsqueda masiva de tránsitos con avance guardado")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sector", type=int, help="Usar las estrellas observadas por TESS en ese sector")
    g.add_argument("--lista", help="Archivo con un TIC o nombre por línea")
    ap.add_argument("--max", type=int, default=100, help="Máximo de estrellas (con --sector). 0 = todas")
    ap.add_argument("--tmag", type=float, nargs=2, metavar=("MIN", "MAX"),
                    help="Rango de magnitud TESS (ej. 9 12). Más brillante = número más chico")
    ap.add_argument("--sin-conocidos", action="store_true",
                    help="Saltar estrellas que ya tienen TOI, CTOI o planeta confirmado")
    ap.add_argument("--semilla", type=int, default=1, help="Para elegir estrellas al azar (reproducible)")
    ap.add_argument("--sectores", type=int, default=1, help="Sectores por estrella (0 = todos)")
    ap.add_argument("--max-senales", type=int, default=3)
    ap.add_argument("--procesos", type=int, default=1, help="Estrellas en paralelo (1-4 según tu PC)")
    ap.add_argument("--tiempo-max", type=int, default=900, help="Segundos máximos por estrella")
    ap.add_argument("--salida", default="resultados_masivo")
    ap.add_argument("--clasificador", default=os.path.join(AQUI, "ml", "clasificador_toi.joblib"))
    ap.add_argument("--catalogos", default=os.path.join(AQUI, "ml", "datos"))
    ap.add_argument("--actualizar-catalogos", action="store_true", help="Descargar catálogos frescos")
    ap.add_argument("--conservar-datos", action="store_true",
                    help="No borrar las curvas de luz descargadas (ocupan ~2 MB por sector)")
    ap.add_argument("--reintentar-errores", action="store_true")
    ap.add_argument("--solo-resumen", action="store_true")
    ap.add_argument("--invertir", action="store_true",
                    help="PRUEBA NULA: invierte el brillo (no puede haber tránsitos). Todo 'candidato' "
                         "que salga así es una falsa alarma: sirve para medir la tasa real de ellas")
    a = ap.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    conocidos = cargar_catalogos(a.catalogos, a.actualizar_catalogos)
    log(f"Catálogos: {len(conocidos)} estrellas con objetos conocidos", a.salida)
    if a.solo_resumen:
        n, c, nv = escribir_resumen(a.salida, conocidos)
        log(f"Resumen: {n} estrellas, {c} candidatos, {nv} sin registro", a.salida)
        return

    # La lista se guarda la primera vez: al reanudar se usa EXACTAMENTE la misma
    ruta_lista = os.path.join(a.salida, "lista_objetivos.txt")
    if os.path.exists(ruta_lista):
        objetivos = lista_desde_archivo(ruta_lista)
        log(f"Reanudando con la lista guardada ({len(objetivos)} estrellas)", a.salida)
    else:
        if a.lista:
            objetivos = lista_desde_archivo(a.lista)
        elif a.sector:
            excluir = set(conocidos) if a.sin_conocidos else set()
            objetivos = lista_desde_sector(a.sector, a.salida, a.tmag, a.max, a.semilla, excluir)
        else:
            raise SystemExit("Indica --sector N o --lista archivo.txt")
        if a.lista and a.sin_conocidos:
            objetivos = [o for o in objetivos if tic_de(o) not in conocidos]
        with open(ruta_lista, "w", encoding="utf-8") as fh:
            fh.write("\n".join(objetivos) + "\n")

    hechos = leer_progreso(a.salida)
    pendientes = [o for o in objetivos if o not in hechos
                  or (a.reintentar_errores and hechos[o]["estado"] in ("error", "tiempo_agotado"))]
    log(f"{len(objetivos)} estrellas en la lista · {len(objetivos) - len(pendientes)} ya hechas · "
        f"{len(pendientes)} pendientes", a.salida)

    opciones = dict(sectores=a.sectores, max_senales=a.max_senales,
                    clasificador=a.clasificador if os.path.exists(a.clasificador) else None,
                    graficos=os.path.join(a.salida, "graficos"), borrar_datos=not a.conservar_datos,
                    invertir=a.invertir)
    if not opciones["clasificador"]:
        log("Aviso: no encontré el modelo de IA; sigo sin probabilidad de IA", a.salida)

    ctx = mp.get_context("spawn")               # igual en Windows, Mac y Linux
    activos = {}                                # objetivo -> (proceso, cola, inicio)
    hechos_ahora, t0 = 0, time.time()
    try:
        while pendientes or activos:
            while pendientes and len(activos) < max(1, a.procesos):
                o = pendientes.pop(0)
                cola = ctx.Queue()
                p = ctx.Process(target=trabajador, args=(o, opciones, cola), daemon=True)
                p.start()
                activos[o] = (p, cola, time.time())
            for o, (p, cola, ini) in list(activos.items()):
                res = None
                try:
                    res = cola.get_nowait()
                except queue.Empty:
                    if time.time() - ini > a.tiempo_max:
                        p.kill()
                        res = {"objetivo": o, "estado": "tiempo_agotado", "senales": [],
                               "segundos": a.tiempo_max}
                    elif not p.is_alive() and p.exitcode not in (0, None):
                        res = {"objetivo": o, "estado": "error", "senales": [],
                               "error": f"el proceso terminó con código {p.exitcode}"}
                if res is None:
                    continue
                p.join(timeout=5)
                del activos[o]
                guardar_progreso(a.salida, res)
                hechos_ahora += 1
                cands = [s for s in res["senales"] if s["veredicto"] == "CANDIDATO PLANETARIO"]
                marca = ""
                for s in cands:
                    estado, _ = cruzar(res.get("tic") or 0, s["periodo"], conocidos)
                    ia = f" IA {s['prob_ia']:.0%}" if s["prob_ia"] is not None else ""
                    marca += f" | candidato P={s['periodo']:.3f} d{ia} → {estado}"
                vel = (time.time() - t0) / hechos_ahora
                log(f"[{hechos_ahora}/{hechos_ahora + len(pendientes) + len(activos)}] {o}: "
                    f"{res['estado']} ({res.get('segundos', 0):.0f} s){marca} · "
                    f"faltan ~{vel * (len(pendientes) + len(activos)) / 60:.0f} min", a.salida)
                if hechos_ahora % 10 == 0:
                    escribir_resumen(a.salida, conocidos)
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("Detenido por el usuario. El avance quedó guardado: vuelve a ejecutar el mismo "
            "comando para continuar.", a.salida)
        for p, _, _ in activos.values():
            p.kill()
    n, c, nv = escribir_resumen(a.salida, conocidos)
    log(f"Listo: {n} estrellas procesadas, {c} candidatos, {nv} sin registro en catálogos. "
        f"Ver {os.path.join(a.salida, 'resumen.md')}", a.salida)


if __name__ == "__main__":
    main()
