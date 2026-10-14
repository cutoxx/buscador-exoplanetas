"""
pruebas_validacion.py
---------------------
Batería estadística para medir errores y sesgos de buscador_transitos.py
con estrellas simuladas donde SABEMOS la respuesta correcta.

  A) Falsas alarmas: estrellas SIN planeta (ruido, ruido rojo, variabilidad,
     rotadores rápidos, cadencias de 2/10/30 min). Ningún "candidato" debería salir.
  B) Inyección-recuperación: planetas de profundidad y período al azar.
     Mide qué fracción se recupera según su SNR (completitud) y si hay sesgo
     por período, y qué tan bien se mide la profundidad.
  C) Binarias eclipsantes: circulares, excéntricas, gemelas y rasantes (V).
     Mide qué fracción se descarta correctamente.
  D) Casos de regresión que fallaban en v1.

Uso:  python pruebas_validacion.py [--rapido]
"""
import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import lightkurve as lk

warnings.filterwarnings("ignore")
import buscador_transitos as bt  # noqa: E402

G, RSOL, MSOL = bt.G_CGS, bt.R_SOL_CM, bt.M_SOL_G


# ---------------------------------------------------------------------------
# Generador de curvas de luz
# ---------------------------------------------------------------------------
def curva_base(rng, cad_min, ruido_ppm, rojo_frac, var_amp, var_per, dias=27.0):
    t = np.arange(0, dias, cad_min / 1440)
    t = t[(t < dias / 2 - 0.5) | (t > dias / 2 + 0.5)]
    fase = rng.uniform(0, 2 * np.pi)
    f = 1 + var_amp * (np.sin(2 * np.pi * t / var_per + fase)
                       + 0.4 * np.sin(4 * np.pi * t / var_per + 2 * fase))
    ruido = rng.normal(0, ruido_ppm * 1e-6, t.size)
    if rojo_frac > 0:
        escala_h = rng.uniform(0.5, 3.0)
        n = max(1, int(round(escala_h * 60 / cad_min)))
        rojo = np.convolve(rng.normal(0, 1, t.size + n), np.ones(n) / n, mode="same")[: t.size]
        ruido += rojo / rojo.std() * rojo_frac * ruido_ppm * 1e-6
    curva_base.ruido = ruido          # se guarda para calcular el SNR esperado real
    return t, f + ruido


def t14_sol(P, b, k=0.05):
    a = (G * MSOL * (P * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / RSOL
    return P / np.pi * np.arcsin(np.sqrt((1 + k) ** 2 - b ** 2) / a) * 24   # horas


def eclipse_v(t, P, t0, prof, dur_h):
    x = np.abs(bt.fase_centrada(t, P, t0)) / (dur_h / 48)
    return 1 - prof * np.clip(1 - x, 0, 1)


def parametros_ruido(rng, permitir_rapidos=True):
    return dict(cad_min=float(rng.choice([2, 2, 2, 10, 30])),
                ruido_ppm=float(np.exp(rng.uniform(np.log(300), np.log(2000)))),
                rojo_frac=float(rng.uniform(0, 0.6)),
                var_amp=float(np.exp(rng.uniform(np.log(1e-4), np.log(2e-2)))),
                var_per=float(np.exp(rng.uniform(np.log(0.3 if permitir_rapidos else 1.0), np.log(15)))))


def correr(lc, max_senales=2):
    senales, _, _ = bt.analizar(lc, bt.Estrella(), max_senales=max_senales, verbose=False)
    return [dict(P=s.periodo, t0=s.t0, dur_h=s.duracion_h, prof_c=s.profundidad_central_ppm,
                 sde=s.sde, snr=s.snr, ver=s.veredicto, motivos=s.motivos,
                 pruebas={k: float(v) for k, v in s.pruebas.items() if not isinstance(v, str)}) for s in senales]


# ---------------------------------------------------------------------------
# Casos
# ---------------------------------------------------------------------------
def caso_nulo(semilla):
    rng = np.random.default_rng(1000 + semilla)
    p = parametros_ruido(rng)
    t, f = curva_base(rng, **p)
    r = correr(lk.LightCurve(time=t, flux=f))
    return dict(tipo="nulo", semilla=semilla, params=p, senales=r)


def caso_inyeccion(semilla):
    rng = np.random.default_rng(2000 + semilla)
    p = parametros_ruido(rng, permitir_rapidos=False)
    p["var_amp"] = min(p["var_amp"], 5e-3)
    t, f = curva_base(rng, **p)
    P = float(np.exp(rng.uniform(np.log(0.7), np.log(13))))
    b = float(rng.uniform(0, 0.85))
    prof = float(np.exp(rng.uniform(np.log(200), np.log(6000)))) * 1e-6
    dur_h = float(t14_sol(P, b, np.sqrt(prof)))
    t0 = float(rng.uniform(0, P))
    f = f * bt.modelo_transito(t, P, t0, prof, dur_h)
    n_in = np.sum(np.abs(bt.fase_centrada(t, P, t0)) < dur_h / 48)
    # SNR esperado REAL: ruido (blanco + rojo) promediado en bloques del largo
    # del tránsito, por la raíz del número de tránsitos con datos.
    ruido, dur = curva_base.ruido, dur_h / 24
    bloques = np.floor(t / dur).astype(int)
    cuenta = np.bincount(bloques)
    medias = np.bincount(bloques, weights=ruido)[cuenta > 0.5 * cuenta.max()] / cuenta[cuenta > 0.5 * cuenta.max()]
    dentro = np.abs(bt.fase_centrada(t, P, t0)) < dur / 2
    n_tr = len(np.unique(np.round((t[dentro] - t0) / P)))
    prof_media = np.mean(1 - bt.modelo_transito(t[dentro], P, t0, prof, dur_h))
    snr_esperado = prof_media / np.std(medias) * np.sqrt(n_tr)
    r = correr(lk.LightCurve(time=t, flux=f))
    return dict(tipo="inyeccion", semilla=semilla, params=p,
                verdad=dict(P=P, t0=t0, prof=prof * 1e6, dur_h=dur_h, b=b,
                            snr_esperado=float(snr_esperado), n_in=int(n_in)),
                senales=r)


def caso_binaria(semilla):
    rng = np.random.default_rng(3000 + semilla)
    p = parametros_ruido(rng, permitir_rapidos=False)
    p["var_amp"] = min(p["var_amp"], 3e-3)
    t, f = curva_base(rng, **p)
    clase = ["circular", "excentrica", "gemelas", "rasante", "gemelas_rasantes"][semilla % 5]
    P = float(np.exp(rng.uniform(np.log(1.0), np.log(10))))
    t0 = float(rng.uniform(0, P))
    prof1 = float(np.exp(rng.uniform(np.log(0.005), np.log(0.2))))
    dur_h = float(t14_sol(P, 0.3, 0.8) * rng.uniform(0.8, 1.2))
    if clase == "circular":
        prof2, fase2, dur2 = prof1 * rng.uniform(0.1, 0.7), 0.5, dur_h
    elif clase == "excentrica":
        prof2 = prof1 * rng.uniform(0.2, 0.8)
        fase2 = float(rng.choice([rng.uniform(0.3, 0.42), rng.uniform(0.58, 0.7)]))
        dur2 = dur_h * rng.uniform(0.6, 1.4)
    elif clase == "gemelas":   # dos estrellas iguales, eclipse de frente: muy profundo
        prof1 = float(rng.uniform(0.10, 0.45))
        prof2, fase2, dur2 = prof1 * rng.uniform(0.9, 1.0), 0.5, dur_h
    elif clase == "gemelas_rasantes":   # iguales y rozándose: poco profundo y en V
        prof1 = float(np.exp(rng.uniform(np.log(0.003), np.log(0.03))))
        prof2, fase2, dur2 = prof1 * rng.uniform(0.9, 1.0), 0.5, dur_h * 0.6
        dur_h *= 0.6
    else:  # rasante: eclipses en V, poco profundos, secundario débil o nulo
        prof1 = float(np.exp(rng.uniform(np.log(0.002), np.log(0.02))))
        prof2, fase2, dur2 = prof1 * rng.uniform(0.0, 0.5), 0.5, dur_h * 0.6
        dur_h *= 0.6
    if clase in ("rasante", "gemelas_rasantes"):
        f = f * eclipse_v(t, P, t0, prof1, dur_h) * eclipse_v(t, P, t0 + fase2 * P, prof2, dur2)
    else:
        f = f * bt.modelo_transito(t, P, t0, prof1, dur_h)
        f = f * bt.modelo_transito(t, P, t0 + fase2 * P, prof2, dur2)
    r = correr(lk.LightCurve(time=t, flux=f), max_senales=3)
    return dict(tipo="binaria", semilla=semilla, clase=clase, params=p,
                verdad=dict(P=P, prof1=prof1 * 1e6, prof2=prof2 * 1e6, fase2=fase2, dur_h=dur_h),
                senales=r)


def caso_regresion(nombre):
    rng = np.random.default_rng(99)
    if nombre == "30min_variable":     # v1 confundía la variabilidad con un planeta
        t, f = curva_base(rng, 30, 700, 0.3, 0.01, 3.0)
        verdad = dict(P=4.1, t0=1.0, prof=3000, dur_h=3.0)
    else:                              # "periodo_20d": v1 devolvía la mitad (10 d)
        t, f = curva_base(rng, 2, 1000, 0.3, 0.002, 8.0)
        verdad = dict(P=20.0, t0=4.0, prof=5000, dur_h=7.0)
    f = f * bt.modelo_transito(t, verdad["P"], verdad["t0"], verdad["prof"] * 1e-6, verdad["dur_h"])
    return dict(tipo="regresion", nombre=nombre, verdad=verdad,
                senales=correr(lk.LightCurve(time=t, flux=f)))


def ejecutar(tarea):
    tipo, arg = tarea
    try:
        return {"nulo": caso_nulo, "inyeccion": caso_inyeccion,
                "binaria": caso_binaria, "regresion": caso_regresion}[tipo](arg)
    except Exception as e:  # registrar el error en vez de esconderlo
        return dict(tipo=tipo, semilla=arg, error=repr(e))


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
def coincide(P_hallado, P_real):
    r = P_hallado / P_real
    if abs(r - 1) < 0.005:
        return "exacto"
    for k in (2, 3, 0.5, 1 / 3):
        if abs(r / k - 1) < 0.005:
            return "alias"
    return None


def resumir(res):
    out = {}
    errores = [r for r in res if "error" in r]
    print(f"\nErrores de ejecución: {len(errores)}")
    for e in errores:
        print("  ", e)

    nulos = [r for r in res if r["tipo"] == "nulo" and "error" not in r]
    fa = [r for r in nulos if any(s["ver"] == "CANDIDATO PLANETARIO" for s in r["senales"])]
    sde = np.array([r["senales"][0]["sde"] for r in nulos])
    print(f"\nA) FALSAS ALARMAS: {len(fa)} de {len(nulos)} estrellas sin planeta dieron 'candidato'")
    print(f"   SDE del pico más alto en ruido: mediana {np.median(sde):.1f}, "
          f"p95 {np.percentile(sde, 95):.1f}, máx {sde.max():.1f}")
    for r in fa:
        s = [x for x in r["senales"] if x["ver"] == "CANDIDATO PLANETARIO"][0]
        print(f"   ! semilla {r['semilla']} {r['params']} -> P={s['P']:.3f} SDE={s['sde']:.1f} SNR={s['snr']:.1f}")
    out["falsas_alarmas"] = (len(fa), len(nulos))
    out["sde_nulo"] = sde.tolist()

    iny = [r for r in res if r["tipo"] == "inyeccion" and "error" not in r]
    filas = []
    for r in iny:
        v = r["verdad"]
        hit = None
        for s in r["senales"]:
            c = coincide(s["P"], v["P"])
            if c:
                hit = (c, s)
                break
        filas.append((v["snr_esperado"], v["P"], v["prof"], hit))
    print(f"\nB) INYECCIÓN-RECUPERACIÓN ({len(filas)} planetas)")
    print("   SNR esperado   n   recuperado+candidato   solo detectado   perdido")
    bordes = [0, 7, 10, 15, 25, 50, 1e9]
    tabla = []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        g = [x for x in filas if lo <= x[0] < hi]
        if not g:
            continue
        cand = sum(1 for x in g if x[3] and x[3][0] == "exacto" and x[3][1]["ver"] == "CANDIDATO PLANETARIO")
        det = sum(1 for x in g if x[3]) - cand
        tabla.append((lo, hi, len(g), cand, det))
        print(f"   {lo:>5g}–{hi if hi < 1e9 else '∞':<6}   {len(g):>3}   {cand:>3} ({100*cand/len(g):3.0f}%)"
              f"            {det:>3}            {len(g)-cand-det:>3}")
    out["completitud"] = tabla
    fuertes = [x for x in filas if x[0] >= 15]
    for nombre, cond in (("P < 4 d", lambda x: x[1] < 4), ("P ≥ 4 d", lambda x: x[1] >= 4)):
        g = [x for x in fuertes if cond(x)]
        if g:
            ok = sum(1 for x in g if x[3] and x[3][1]["ver"] == "CANDIDATO PLANETARIO")
            print(f"   Con SNR≥15, {nombre}: {ok}/{len(g)} candidatos ({100*ok/len(g):.0f}%)")
    razones = [x[3][1]["prof_c"] / x[2] for x in filas if x[3] and x[3][0] == "exacto" and x[0] >= 15]
    if razones:
        print(f"   Profundidad medida / real (SNR≥15): mediana {np.median(razones):.3f}, "
              f"rango 16–84%: {np.percentile(razones,16):.2f}–{np.percentile(razones,84):.2f}")
        out["razon_profundidad"] = razones
    malos = [x for x in filas if x[0] >= 15 and not (x[3] and x[3][1]["ver"] == "CANDIDATO PLANETARIO")]
    for x in malos:
        r = [r for r in iny if r["verdad"]["P"] == x[1]][0]
        info = (x[3][1]["ver"], x[3][1]["motivos"]) if x[3] else ("no recuperado",
                [(round(s["P"], 3), s["ver"]) for s in r["senales"]])
        print(f"   ! SNR {x[0]:.0f} P={x[1]:.2f} prof={x[2]:.0f}ppm sem {r['semilla']}: {info}")

    eb = [r for r in res if r["tipo"] == "binaria" and "error" not in r]
    print(f"\nC) BINARIAS ECLIPSANTES ({len(eb)})")
    for clase in ("circular", "excentrica", "gemelas", "rasante", "gemelas_rasantes"):
        g = [r for r in eb if r["clase"] == clase]
        pasa = [r for r in g if any(s["ver"] == "CANDIDATO PLANETARIO" for s in r["senales"])]
        adv = [r for r in pasa if any("forma de V" in m for s in r["senales"] for m in s["motivos"])]
        print(f"   {clase:11s}: {len(g)-len(pasa)}/{len(g)} descartadas"
              f"  (de las que pasaron, {len(adv)} con advertencia de forma en V)")
        for r in pasa:
            s = [x for x in r["senales"] if x["ver"] == "CANDIDATO PLANETARIO"][0]
            print(f"      · pasó: {r['verdad']} -> P={s['P']:.3f} pruebas={s['pruebas']}")
    out["binarias"] = {c: [r["senales"] for r in eb if r["clase"] == c] for c in ("circular",)}

    print("\nD) REGRESIONES (casos que fallaban en v1)")
    for r in [r for r in res if r["tipo"] == "regresion" and "error" not in r]:
        s = r["senales"][0]
        c = coincide(s["P"], r["verdad"]["P"])
        print(f"   {r['nombre']:15s} real P={r['verdad']['P']} -> P={s['P']:.4f} ({c or 'MAL'}), "
              f"{s['ver']}, prof {s['prof_c']:.0f}/{r['verdad']['prof']} ppm")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--solo", default=None, help="nulo,inyeccion,binaria,regresion")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    n_nulo, n_iny, n_eb = (12, 16, 8) if a.rapido else (80, 120, 40)
    if a.n:
        n_nulo = n_iny = n_eb = a.n
    tareas = []
    tipos = a.solo.split(",") if a.solo else ["nulo", "inyeccion", "binaria", "regresion"]
    if "regresion" in tipos:
        tareas += [("regresion", "30min_variable"), ("regresion", "periodo_20d")]
    if "nulo" in tipos:
        tareas += [("nulo", i) for i in range(n_nulo)]
    if "inyeccion" in tipos:
        tareas += [("inyeccion", i) for i in range(n_iny)]
    if "binaria" in tipos:
        tareas += [("binaria", i) for i in range(n_eb)]
    t_ini = time.time()
    with Pool(2) as pool:
        res = []
        for k, r in enumerate(pool.imap_unordered(ejecutar, tareas), 1):
            res.append(r)
            if k % 10 == 0:
                print(f"  {k}/{len(tareas)} ({time.time()-t_ini:.0f} s)", file=sys.stderr, flush=True)
    with open("resultados_validacion.json", "w") as fh:
        json.dump(res, fh, default=float)
    resumir(res)
    print(f"\nTiempo total: {time.time()-t_ini:.0f} s")
