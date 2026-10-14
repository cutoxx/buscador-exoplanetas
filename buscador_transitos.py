"""
buscador_transitos.py  (v2)
===========================
Busca planetas en la curva de luz (light curve) de una estrella con el método de
tránsito (transit method) y el algoritmo BLS (Box Least Squares).

Para cada señal NO dice "es un planeta": la clasifica y explica por qué.
  * SIN DETECCIÓN           -> la señal no supera el umbral de significancia
  * CANDIDATO PLANETARIO    -> pasa todas las pruebas (igual requiere confirmación)
  * PROBABLE FALSO POSITIVO -> binaria eclipsante, variabilidad estelar, etc.
  * DATOS INSUFICIENTES     -> menos de 2 tránsitos con datos, o no hay zonas
                               libres suficientes para medir el ruido

Uso:
    python buscador_transitos.py --demo planeta     # también: multi, binaria, ruido
    python buscador_transitos.py --estrella "Pi Mensae"
    python buscador_transitos.py --estrella "TOI 270" --sectores 3
    python buscador_transitos.py --estrella "TIC 261136679" --sectores 0   # 0 = todos

Instalación (recomendado en un entorno virtual):
    python -m venv venv && source venv/bin/activate
    pip install lightkurve astropy numpy scipy matplotlib
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.ndimage import median_filter
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constantes y umbrales (calibrados con pruebas_validacion.py)
# ---------------------------------------------------------------------------
UMBRAL_SDE = 10.0        # significancia del pico del periodograma (ver calibración)
UMBRAL_SNR = 7.1         # señal/ruido mínimo (estándar de Kepler/TESS)
MIN_TRANSITOS = 3        # como Kepler: con 2 eventos cualquiera "inventa" un período
SNR_FUERTE = 10.0        # bajo esto, en datos reales de TESS muchas señales resultan falsas
UMBRAL_SIGMA_TEST = 3.0  # sobre esto, una prueba levanta una ADVERTENCIA
# Para DESCARTAR se exige más: se aplican varias pruebas a cada señal y el ruido
# real de TESS tiene colas pesadas; con 3σ, planetas reales (ej. Pi Mensae c
# con 3 sectores) fallaban alguna por mala suerte. Kepler también era más exigente.
UMBRAL_FP_SECUNDARIO = 5.0
UMBRAL_FP_PAR_IMPAR = 4.0
FACTOR_LIMBO = 0.85      # (Rp/R*)^2 ≈ 0.85 x profundidad central (oscurecimiento al limbo típico)
VARIABILIDAD_EXTREMA = 0.10  # rango de brillo (>10 %) sobre el cual los resultados son poco confiables
PISO_SISTEMATICO = 0.07  # variación aceptada entre tránsitos (cadencia, manchas)
UMBRAL_FORMA_V = 0.25    # fracción de fondo plano bajo la cual el tránsito parece una V
MAX_PERIODOS = 250_000   # límite de la grilla para no tardar horas
R_SOL_EN_TIERRAS = 109.2
R_JUP_EN_TIERRAS = 11.2
G_CGS = 6.674e-8
R_SOL_CM = 6.957e10
M_SOL_G = 1.989e33


@dataclass
class Estrella:
    radio_sol: float = 1.0      # radio en radios solares
    masa_sol: float = 1.0       # masa en masas solares
    supuesta: bool = True       # True = no había datos, se asumió el Sol
    teff: float = float("nan")  # temperatura (K), si el catálogo la tiene
    logg: float = float("nan")  # gravedad superficial (log g, cgs)


@dataclass
class Senal:
    numero: int
    periodo: float
    t0: float
    duracion_h: float
    profundidad_ppm: float          # profundidad media del tránsito (BLS)
    profundidad_central_ppm: float  # profundidad en el centro (para el radio)
    sde: float
    snr: float
    n_transitos: int
    radio_tierras: float
    veredicto: str = ""
    motivos: list = field(default_factory=list)
    pruebas: dict = field(default_factory=dict)
    secundario_de: int = 0          # >0: esta señal es el 2º eclipse de esa señal
    prob_planeta: float = float("nan")  # probabilidad según el modelo de IA (opcional)


# ---------------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------------
def descargar(nombre: str, sectores: int, autor: str | None):
    """Descarga curvas de luz de TESS. Prueba primero las de 2 min (SPOC) y, si
    no hay, las de imágenes completas (FFI). Así no se excluyen estrellas que
    nunca tuvieron observación de 2 min (que son la mayoría)."""
    autores = [autor] if autor else ["SPOC", "TESS-SPOC", "QLP"]
    for a in autores:
        res = lk.search_lightcurve(nombre, mission="TESS", author=a)
        if len(res) == 0:
            continue
        if a == "SPOC":  # en SPOC preferimos 120 s (hay también 20 s, más pesado)
            r120 = res[np.isclose(res.exptime.value, 120)]
            res = r120 if len(r120) else res
        # Orden numérico por sector (como texto, "Sector 10" quedaría antes que "Sector 2")
        res = res[np.argsort(np.asarray(res.table["sequence_number"], dtype=int), kind="stable")]
        if sectores > 0:
            res = res[:sectores]
        print(f"Datos: {len(res)} sector(es) de {a} -> {', '.join(res.mission)}")
        coleccion = res.download_all(quality_bitmask="default")
        lc = coleccion.stitch()               # une y normaliza cada sector
        return lc, a
    raise SystemExit(f"No encontré curvas de luz de TESS para '{nombre}'.")


def parametros_estrella(lc) -> Estrella:
    """Lee radio y gravedad de la estrella del encabezado (catálogo TIC)."""
    m = lc.meta
    radio = m.get("RADIUS")
    logg = m.get("LOGG")
    try:
        radio = float(radio)
        logg = float(logg)
        if not (np.isfinite(radio) and np.isfinite(logg)) or radio <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return Estrella()
    masa = (10 ** logg) * (radio * R_SOL_CM) ** 2 / G_CGS / M_SOL_G
    try:
        teff = float(m.get("TEFF"))
    except (TypeError, ValueError):
        teff = float("nan")
    return Estrella(radio_sol=radio, masa_sol=float(np.clip(masa, 0.08, 10)), supuesta=False,
                    teff=teff, logg=logg)


def a_arreglo_simple(lc):
    """Los datos reales de TESS vienen como arreglos "enmascarados" de astropy;
    se pasan a arreglos normales (NaN donde el dato es inválido) para que
    todos los cálculos funcionen igual que con datos simulados."""
    def simple(x):
        v = getattr(x, "value", x)
        m = np.asarray(getattr(v, "mask", np.zeros(np.shape(v), dtype=bool)))
        v = np.array(getattr(v, "unmasked", v), dtype=float)
        v[m] = np.nan
        return v
    t = simple(lc.time.value if hasattr(lc.time, "value") else lc.time)
    f = simple(lc.flux)
    e = simple(lc.flux_err) if getattr(lc, "flux_err", None) is not None else np.full_like(f, np.nan)
    ok = np.isfinite(t) & np.isfinite(f)
    # Puntos físicamente imposibles: una caída de más del 70% no la produce
    # ningún planeta ni una binaria de dos estrellas iguales (máx. ~50%).
    # Aparecen en algunos sectores de QLP como luz medida casi en cero.
    med = np.nanmedian(f[ok]) if ok.any() else np.nan
    if np.isfinite(med) and med > 0:
        imposibles = ok & ((f < 0.3 * med) | (f > 2.0 * med))
        if imposibles.any():
            print(f"Aviso: se descartan {int(imposibles.sum())} puntos imposibles "
                  f"(luz bajo 30% o sobre 200% de lo normal; datos dañados).")
            ok &= ~imposibles
    nuevo = lk.LightCurve(time=t[ok], flux=f[ok], flux_err=e[ok])
    nuevo.meta.update({k: v for k, v in lc.meta.items() if k in ("RADIUS", "LOGG", "TEFF", "TICID", "OBJECT")})
    return nuevo


def cadencia_dias(t) -> float:
    return float(np.nanmedian(np.diff(t)))


# ---------------------------------------------------------------------------
# 2. LIMPIEZA
# ---------------------------------------------------------------------------
def tendencia_biweight(t, f, ventana, excluir=None, c=5.0, iteraciones=5):
    """Tendencia con un promedio robusto (biweight de Tukey) en una ventana
    móvil de `ventana` días. A diferencia de un filtro común (Savitzky-Golay,
    que usaba la v1), el biweight IGNORA lo que se sale de lo normal: no se
    "come" los tránsitos ni fabrica caídas falsas al lado de una subida.
    Recomendado por Hippke et al. 2019 (paquete wotan)."""
    usar = np.ones(t.size, dtype=bool) if excluir is None else ~excluir
    tu, fu = t[usar], f[usar]
    lo = np.searchsorted(tu, t - ventana / 2)
    hi = np.searchsorted(tu, t + ventana / 2)
    tend = np.full(t.size, np.nan)
    for i in range(t.size):
        x = fu[lo[i]:hi[i]]
        if x.size < 5:
            continue
        M = np.median(x)
        for _ in range(iteraciones):
            mad = np.median(np.abs(x - M))
            if mad == 0:
                break
            u = (x - M) / (c * mad)
            w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
            M_nuevo = np.sum(w * x) / np.sum(w)
            if abs(M_nuevo - M) < 1e-8:
                M = M_nuevo
                break
            M = M_nuevo
        tend[i] = M
    malos = ~np.isfinite(tend)
    if malos.any() and (~malos).sum() >= 2:
        tend[malos] = np.interp(t[malos], t[~malos], tend[~malos])
    return tend


def aplanar(lc, ventana_dias: float, mascara=None):
    """Quita la variación lenta de la estrella (manchas, rotación, deriva).

    - Ventana en DÍAS (v1 usaba 901 puntos fijos: 30 h con datos de 2 min,
      pero 19 DÍAS con datos de 30 min, y no limpiaba nada).
    - Promedio robusto (biweight): no lo afectan tránsitos ni subidas.
    - mascara: puntos (tránsitos ya encontrados) que no se usan para estimar
      la tendencia; ahí la tendencia se calcula con los vecinos.
    - Outliers: solo se eliminan los que suben (llamaradas, rayos cósmicos).
      Nunca los que bajan, porque esos pueden ser justamente el tránsito.
    """
    lc = lc.remove_nans().normalize()
    t, f = lc.time.value, lc.flux.value
    if mascara is not None and mascara.mean() > 0.7:
        # Una "señal" que tapa casi todo es ruido con duración absurda: no la protegemos
        mascara = None
    tend = tendencia_biweight(t, f, ventana_dias, mascara)
    plano = lc.copy()
    plano.flux = f / tend
    if plano.flux_err is not None:
        plano.flux_err = lc.flux_err.value / tend
    plano = plano.remove_nans()
    return plano.remove_outliers(sigma_lower=np.inf, sigma_upper=4)


def quitar_rotacion(lc, max_periodos=2, n_armonicos=4, verbose=False):
    """Pre-blanqueo (prewhitening): si la estrella varía de forma periódica
    (manchas que rotan), se detecta el período con Lomb-Scargle y se resta un
    modelo de ondas (fundamental + armónicos), ajustado por tramos porque las
    manchas cambian con el tiempo. Los puntos bajos (posibles tránsitos) se
    ignoran en el ajuste, así el modelo no "aprende" el tránsito.
    Sin esto, en estrellas de rotación rápida la ventana de limpieza no
    alcanza a seguir la variación y los planetas se pierden (sesgo contra
    estrellas jóvenes/activas)."""
    from astropy.timeseries import LombScargle
    lc = lc.remove_nans().normalize()
    t, f = lc.time.value, lc.flux.value.copy()
    cad = cadencia_dias(t)
    info = []
    for _ in range(max_periodos):
        paso = max(1, int(round((30 / 1440) / cad)))           # LS sobre datos de ~30 min
        frec, pot = LombScargle(t[::paso], f[::paso]).autopower(
            minimum_frequency=1 / 15, maximum_frequency=1 / 0.1, samples_per_peak=10)
        Pv = 1 / frec[np.argmax(pot)]
        modelo = np.ones_like(f)
        # Tramos PAREJOS de al menos 3 ciclos (un tramo final corto, con tantos
        # parámetros, se "comería" un tránsito). Armónicos con período >= 6 h,
        # para no imitar la forma de un tránsito.
        span = t.max() - t.min()
        n_tramos = max(1, int(span // max(3 * Pv, 3.0)))
        bordes = np.linspace(t.min(), t.max() + 1e-9, n_tramos + 1)
        n_arm = int(np.clip(np.floor(Pv / 0.25), 1, n_armonicos))
        for t_ini, t_fin in zip(bordes[:-1], bordes[1:]):
            sel = (t >= t_ini) & (t < t_fin)
            if sel.sum() < 50:
                continue
            x = t[sel] - t[sel].mean()
            cols = [np.ones_like(x), x]
            for n in range(1, n_arm + 1):
                cols += [np.sin(2 * np.pi * n * x / Pv), np.cos(2 * np.pi * n * x / Pv)]
            A = np.vstack(cols).T
            usar = np.ones(sel.sum(), dtype=bool)
            for _it in range(4):                                # recorte asimétrico
                coef, *_ = np.linalg.lstsq(A[usar], f[sel][usar], rcond=None)
                res = f[sel] - A @ coef
                s = ruido_robusto(res[usar])
                usar = (res > -2.5 * s) & (res < 4 * s)
            modelo[sel] = A @ coef
        mejora = ruido_robusto(f) / ruido_robusto(f / modelo)
        if mejora < 1.15:                                       # no había rotación clara
            break
        f = f / modelo
        info.append((round(float(Pv), 3), round(float(mejora), 2)))
    if verbose and info:
        print("Variabilidad periódica restada (período d, mejora del ruido x):", info)
    salida = lc.copy()
    salida.flux = f * lc.flux.unit if hasattr(lc.flux, "unit") else f
    return salida, info


def elegir_ventana(lc, opciones=(0.5, 0.75, 1.0)):
    """Elige la ventana de limpieza según la estrella: la MÁS LARGA cuyo ruido
    residual (en bloques de 2 h) no supere en 10 % al mejor. Estrellas
    tranquilas -> ventana larga (protege tránsitos largos); estrellas que
    varían rápido -> ventana corta (si no, la variabilidad tapa al planeta)."""
    ruidos = []
    for w in opciones:
        pl = aplanar(lc, w)
        t, f = pl.time.value, pl.flux.value
        bloques = np.floor((t - t.min()) / (2 / 24)).astype(int)
        cuenta = np.bincount(bloques)
        ok = cuenta >= 0.5 * cuenta.max()
        ruidos.append(ruido_robusto(np.bincount(bloques, weights=f)[ok] / cuenta[ok]))
    ruidos = np.array(ruidos)
    validas = [w for w, r in zip(opciones, ruidos) if r <= 1.2 * ruidos.min()]
    return max(validas), {w: round(float(r) * 1e6, 1) for w, r in zip(opciones, ruidos)}


def ruido_robusto(f) -> float:
    """Desviación estándar resistente a outliers (MAD * 1.4826)."""
    return float(1.4826 * np.nanmedian(np.abs(f - np.nanmedian(f))))


def mascara_transitos(t, senales):
    m = np.zeros(t.size, dtype=bool)
    for s in senales:
        fase = fase_centrada(t, s.periodo, s.t0)
        m |= np.abs(fase) < 0.75 * s.duracion_h / 24 + cadencia_dias(t)
    return m


def fase_centrada(t, periodo, t0):
    """Tiempo (días) desde el centro del tránsito más cercano."""
    return ((t - t0 + 0.5 * periodo) % periodo) - 0.5 * periodo


# ---------------------------------------------------------------------------
# 3. BÚSQUEDA
# ---------------------------------------------------------------------------
def duracion_minima(cad):
    return max(0.5 / 24, 3 * cad)                       # al menos 3 puntos por tránsito


def duracion_max_busqueda(periodo, estrella: Estrella):
    """Duración máxima de tránsito que permite la física (3ª ley de Kepler)
    para ese período, CON MARGEN: estrella más grande que la estimada (o
    hasta 2 radios solares si no se conoce), planeta grande y órbita excéntrica.
    Sin este límite, BLS prueba 'tránsitos' de 10 h en órbitas de 12 h y
    confunde variabilidad estelar con planetas."""
    radio = 2.0 if estrella.supuesta else 1.3 * estrella.radio_sol
    masa = 1.0 if estrella.supuesta else estrella.masa_sol
    a = (G_CGS * masa * M_SOL_G * (periodo * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    t14 = periodo / np.pi * np.arcsin(np.minimum(1.0, 1.2 * radio * R_SOL_CM / a))
    return np.minimum(1.3 * t14, 0.45 * periodo)


@dataclass
class Periodograma:
    period: np.ndarray
    power: np.ndarray
    duration: np.ndarray
    transit_time: np.ndarray


def bls_periodograma(t, f, dy, periodo_min, periodo_max, estrella, tramos=10):
    """BLS con grilla UNIFORME EN FRECUENCIA (v1 usaba una grilla uniforme en
    período, que a períodos cortos queda gruesa y se salta señales).
    Se busca por tramos de período, cada uno solo con duraciones físicas."""
    bls = BoxLeastSquares(t, f, dy)
    d_min = duracion_minima(cadencia_dias(t))
    factor = 1.0
    periodos = bls.autoperiod([d_min], minimum_period=periodo_min,
                              maximum_period=periodo_max, frequency_factor=factor)
    if periodos.size > MAX_PERIODOS:
        factor = periodos.size / MAX_PERIODOS
        periodos = bls.autoperiod([d_min], minimum_period=periodo_min,
                                  maximum_period=periodo_max, frequency_factor=factor)
    periodos = np.sort(periodos)
    bordes = np.geomspace(periodos[0], periodos[-1] * 1.0001, tramos + 1)
    partes = []
    for p0, p1 in zip(bordes[:-1], bordes[1:]):
        sel = periodos[(periodos >= p0) & (periodos < p1)]
        if sel.size == 0:
            continue
        d_max = max(float(duracion_max_busqueda(p1, estrella)), 1.2 * d_min)
        d_max = min(d_max, 0.95 * p0)
        duraciones = np.geomspace(d_min, d_max, 10) if d_max > 1.2 * d_min else np.array([d_min])
        r = bls.power(sel, duraciones, objective="likelihood")
        partes.append((sel, np.asarray(r.power), np.asarray(r.duration), np.asarray(r.transit_time)))
    res = Periodograma(*(np.concatenate([p[j] for p in partes]) for j in range(4)))
    # SDE: qué tan alto es el pico respecto al "piso" del periodograma.
    # Se resta una mediana móvil porque el piso cambia con el período.
    k = max(101, (res.period.size // 200) | 1)
    potencia = res.power - median_filter(res.power, size=k, mode="nearest")
    sde = (potencia - np.mean(potencia)) / np.std(potencia)
    return bls, res, sde, factor


def refinar(t, f, dy, periodo, estrella, baseline):
    """Afina período, duración y época con una grilla fina alrededor del pico."""
    d_min = duracion_minima(cadencia_dias(t))
    d_max = max(float(duracion_max_busqueda(periodo, estrella)), 1.2 * d_min)
    duraciones = np.geomspace(d_min, min(d_max, 0.9 * periodo), 40)
    paso = periodo ** 2 * 0.25 * d_min / baseline ** 2
    periodos = np.linspace(periodo - 40 * paso, periodo + 40 * paso, 401)
    periodos = periodos[periodos > duraciones.max() / 0.9]
    bls = BoxLeastSquares(t, f, dy)
    res = bls.power(periodos, duraciones, objective="likelihood")
    i = int(np.argmax(res.power))
    dur = float(res.duration[i])
    if not (np.isfinite(res.power[i]) and dur >= 0.9 * d_min):
        raise ValueError("refinamiento sin solución válida")
    return bls, float(res.period[i]), float(res.transit_time[i]), dur


# ---------------------------------------------------------------------------
# 4. PRUEBAS DE VALIDACIÓN (vetting)
# ---------------------------------------------------------------------------
# Principio: los tránsitos reales y los "tránsitos de mentira" (momentos al
# azar sin nada) se tratan EXACTAMENTE igual: protegidos en el filtro de
# limpieza y medidos con una recta local. Así el error sale de los datos
# mismos (ruido blanco + rojo + lo que el filtro no quitó), sin suponer nada.
# (Una versión anterior protegía solo los reales: el filtro "se comía" parte
# del ruido en el resto y el error salía subestimado -> falsas binarias.)

def profundidad_local(t, f, tc, dur, excluir):
    """Profundidad de UN tránsito centrado en tc, contra una recta ajustada a
    los datos de ambos lados (de 0.75 a 3 duraciones del centro).
    Recta y no parábola: extrapolar una parábola al centro mete mucho ruido.
    Devuelve (profundidad, cobertura) o None."""
    cad = cadencia_dias(t)
    radio_ventana = max(3 * dur, dur + 0.25)
    i0, i1 = np.searchsorted(t, [tc - radio_ventana, tc + radio_ventana])
    tt, ff, ex = t[i0:i1] - tc, f[i0:i1], excluir[i0:i1]
    dentro = (np.abs(tt) < 0.5 * dur) & ~ex
    fuera = (np.abs(tt) > 0.75 * dur) & ~ex
    cobertura = dentro.sum() / (dur / cad)
    if cobertura < 0.5 or np.sum(fuera & (tt < 0)) < 5 or np.sum(fuera & (tt > 0)) < 5:
        return None
    coef = np.polyfit(tt[fuera], ff[fuera], 1)
    base = np.polyval(coef, tt[dentro])
    return float(np.mean((base - ff[dentro]) / base)), float(cobertura)


def medir_serie(t, f, periodo, t0, dur, excluir):
    """Profundidad local de cada época del período. Filas: (época, prof, cobertura)."""
    n0 = int(np.floor((t.min() - t0) / periodo))
    n1 = int(np.ceil((t.max() - t0) / periodo))
    filas = []
    for e in range(n0, n1 + 1):
        r = profundidad_local(t, f, t0 + e * periodo, dur, excluir)
        if r is not None:
            filas.append((e, r[0], r[1]))
    return np.array(filas) if filas else np.zeros((0, 3))


def ventana_tr(t, periodo, t0, dur):
    return np.abs(fase_centrada(t, periodo, t0)) < 0.75 * dur + cadencia_dias(t)


def medir_con_nulo(lc, ventana_dias, previas, P, t0, dur, n_objetivo=150, semilla=0):
    """Mide tránsitos (fase 0) y posible secundario (fase 0.5) y, con el MISMO
    procedimiento, la 'profundidad' en momentos al azar. Devuelve
    (tabla_primario, tabla_secundario, error_de_un_transito, n_nulos)."""
    t_all = lc.time.value
    cad = cadencia_dias(t_all)
    reales = (mascara_transitos(t_all, previas) | ventana_tr(t_all, P, t0, dur)
              | ventana_tr(t_all, P, t0 + P / 2, dur))

    plano = a_10min(aplanar(lc, ventana_dias, mascara=reales))
    tr, fr = plano.time.value, plano.flux.value
    ex = mascara_transitos(tr, previas)
    tab1 = medir_serie(tr, fr, P, t0, dur, ex)
    tab2 = medir_serie(tr, fr, P, t0 + P / 2, dur, ex)

    # Centros falsos: lejos de cualquier tránsito real y separados entre sí
    rng = np.random.default_rng(semilla)
    ancho = 1.5 * dur + 2 * cad
    sep = 1.5 * ancho                        # los falsos no se tocan entre sí
    lejos = ~(np.convolve(reales, np.ones(max(1, int(ancho / cad)) | 1), mode="same") > 0)
    candidatos = t_all[lejos]
    libre = max(0.0, 0.6 - reales.mean())    # tapar como máximo 60 % de los datos
    por_ronda = int(np.clip(libre * (t_all.size * cad) / ancho, 3, 40))
    rondas = int(np.clip(np.ceil(n_objetivo / por_ronda), 2, 8))
    nulos = []
    for _ in range(rondas):
        if candidatos.size == 0:
            break
        centros = []
        for c in rng.permutation(candidatos) + rng.uniform(-cad / 2, cad / 2, candidatos.size):
            if all(abs(c - x) > sep for x in centros):
                centros.append(c)
                if len(centros) >= por_ronda:
                    break
        centros = np.array(centros)
        falsos = np.zeros(t_all.size, dtype=bool)
        for c in centros:
            falsos |= np.abs(t_all - c) < 0.5 * ancho
        pl = a_10min(aplanar(lc, ventana_dias, mascara=reales | falsos))
        tn, fn = pl.time.value, pl.flux.value
        exn = (mascara_transitos(tn, previas) | ventana_tr(tn, P, t0, dur)
               | ventana_tr(tn, P, t0 + P / 2, dur))
        for c in centros:
            r = profundidad_local(tn, fn, c, dur, exn)
            if r is not None:   # mismo criterio que los reales; se lleva a cobertura completa
                nulos.append(r[0] * np.sqrt(min(r[1], 1.0)))
    nulos = np.array(nulos)
    if nulos.size >= 20:
        # El mayor entre el estimador robusto (MAD) y la desviación estándar
        # recortada a 5σ: el ruido real tiene "colas" más pesadas que una
        # campana, y el MAD solo las ignora (error subestimado).
        mad = ruido_robusto(nulos)
        recortados = nulos[np.abs(nulos - np.median(nulos)) < 5 * mad]
        err1 = max(mad, float(np.std(recortados)))
    else:   # muy pocos datos libres: estimación conservadora (ruido x2)
        err1 = 2 * ruido_robusto(fr) / np.sqrt(max(dur / cadencia_dias(tr), 1))
    return tab1, tab2, float(err1), int(nulos.size)


def media_ponderada(d, e):
    w = 1 / e ** 2
    return float(np.sum(w * d) / np.sum(w)), float(1 / np.sqrt(np.sum(w)))


def bordes_de_datos(t, hueco=0.5):
    """Inicio y fin de cada tramo continuo de datos (huecos > `hueco` días)."""
    t = np.sort(t)
    cortes = np.flatnonzero(np.diff(t) > hueco)
    return np.unique(np.r_[t[0], t[cortes], t[cortes + 1], t[-1]])


def evaluar(s: Senal, tab, tab2, err1, n_nulos, bls, estrella: Estrella, previas, bordes=None):
    """Aplica las pruebas y asigna un veredicto. Cada prueba deja su número
    en s.pruebas para revisarlo a mano."""
    P, t0, dur = s.periodo, s.t0, s.duracion_h / 24
    motivos = []
    s.pruebas["error_un_transito_ppm"] = round(err1 * 1e6, 1)
    s.pruebas["n_nulos"] = n_nulos

    # --- 4.1 tránsitos individuales ---------------------------------------------
    s.n_transitos = int(len(tab))
    s.snr = 0.0
    if s.n_transitos:
        e = err1 / np.sqrt(tab[:, 2])           # menos cobertura -> más error
        d_med, d_err = media_ponderada(tab[:, 1], e)
        s.snr = float(d_med / d_err)            # SNR empírico (incluye ruido rojo)
        s.pruebas["profundidad_local_ppm"] = round(d_med * 1e6, 1)
        # Al reiniciar cada órbita, TESS tiene "rampas" (se calienta, luz
        # dispersa): caídas falsas justo al borde de los huecos de datos. Se
        # mide cuánto queda de la señal SIN los tránsitos cerca de un borde.
        if bordes is not None and len(bordes):
            tc = t0 + tab[:, 0] * P
            cerca = np.min(np.abs(tc[:, None] - np.asarray(bordes)[None, :]), axis=1) < 0.5
            s.pruebas["transitos_cerca_borde"] = int(cerca.sum())
            if cerca.any():
                lejos = ~cerca
                s.pruebas["snr_sin_bordes"] = (round(float(np.divide(*media_ponderada(tab[lejos, 1], e[lejos]))), 1)
                                               if lejos.sum() >= 1 else 0.0)
                s.pruebas["transitos_lejos_borde"] = int(lejos.sum())
        # Para comparar tránsitos entre sí se acepta una pequeña variación real
        # (muestreo de la cadencia, manchas): sin esto, a SNR alto cualquier
        # imperfección parece "inconsistente" y se descartan planetas reales.
        e = np.hypot(e, PISO_SISTEMATICO * max(d_med, 0))
    if s.n_transitos >= 2:
        z = (tab[:, 1] - d_med) / e
        dof = s.n_transitos - 1
        s.pruebas["consistencia_z"] = round(float((np.sum(z ** 2) - dof) / np.sqrt(2 * dof)), 2)
        # Tránsito "faltante": >4σ por debajo de la media (medido en sigmas,
        # así no castiga a los planetas chicos, de tránsitos débiles)
        s.pruebas["transitos_faltantes"] = int(np.sum(z < -4))

        # --- 4.2 par/impar: si difieren, es una binaria con período 2P --------
        par, impar = tab[:, 0] % 2 == 0, tab[:, 0] % 2 != 0
        if par.any() and impar.any():
            dp, ep = media_ponderada(tab[par, 1], e[par])
            di, ei = media_ponderada(tab[impar, 1], e[impar])
            s.pruebas["par_impar_sigma"] = round(abs(dp - di) / np.hypot(ep, ei), 2)

    # --- 4.3 eclipse secundario en fase 0.5 -------------------------------------
    if len(tab2):
        d2, e2 = media_ponderada(tab2[:, 1], err1 / np.sqrt(tab2[:, 2]))
        s.pruebas["secundario_ppm"] = round(d2 * 1e6, 1)
        s.pruebas["secundario_sigma"] = round(d2 / e2, 2)

    # --- 4.4 ¿se parece más a una onda (variabilidad) que a una caja? ---------
    # astropy: valor > 0 significa que el modelo de onda es MEJOR que el tránsito
    try:
        st = bls.stats(P, dur, t0)
        s.pruebas["onda_menos_caja"] = round(float(st["harmonic_delta_log_likelihood"]), 1)
    except Exception:
        pass

    # --- 4.5 física: tamaño y duración posibles ---------------------------------
    a_cm = (G_CGS * estrella.masa_sol * M_SOL_G * (P * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    r_cm = estrella.radio_sol * R_SOL_CM
    k = np.sqrt(max(s.profundidad_central_ppm, 0) / 1e6)
    dur_max_h = P * 24 / np.pi * np.arcsin(min(1.0, r_cm * (1 + k) / a_cm))
    s.pruebas["duracion_max_fisica_h"] = round(dur_max_h, 2)
    margen = 2.5 if estrella.supuesta else 1.6  # margen por excentricidad / incerteza

    # --- 4.6 relación con señales anteriores ------------------------------------
    # Ojo: dos planetas reales pueden tener períodos en razón ~2 (resonancia,
    # ej. TOI-270 c y d). Por eso no basta la razón: además los tránsitos nuevos
    # tienen que caer en el mismo lugar (alias) o en la mitad (secundario).
    relacion = None
    for p in previas:
        razon = P / p.periodo
        entero = max(round(razon), round(1 / razon), 1)
        if abs(max(razon, 1 / razon) - entero) > 0.003 * entero or entero > 4:
            continue
        # Fase de una señal medida en la órbita de período MÁS CORTO de las dos:
        # 0 = los eventos coinciden (eco), otro valor = fase fija distinta.
        P_corto = min(P, p.periodo)
        desfase = ((t0 - p.t0) / P_corto) % 1.0
        tol = max(p.duracion_h, s.duracion_h) / 24 / P_corto
        if min(desfase, 1 - desfase) < tol:
            relacion = (p.numero, f"cae sobre los tránsitos de la señal #{p.numero} "
                                  f"(razón {razon:.2f}): es un eco de esa señal, no algo nuevo")
            continue
        # Período igual o múltiplo EXACTO (±0.3 %) y eventos en otra fase fija de
        # la órbita: es el otro eclipse del mismo sistema. (Planetas en resonancia
        # real nunca son exactos: TOI-270 c y d están en 2.010, no en 2.000.)
        d_este = s.pruebas.get("profundidad_local_ppm", s.profundidad_ppm)
        d_prev = p.pruebas.get("profundidad_local_ppm", p.profundidad_ppm)
        prev_en_v = p.pruebas.get("fraccion_plana", 1) <= UMBRAL_FORMA_V
        if (P >= p.periodo and abs(desfase - 0.5) < tol and d_este < 1500
                and d_este < 0.1 * d_prev and not prev_en_v):
            # Misma física que en 4.3: un Júpiter muy caliente (ej. WASP-18 b)
            # muestra su propio eclipse al pasar detrás de la estrella.
            relacion = (p.numero, f"es el eclipse del planeta #{p.numero} al pasar DETRÁS de "
                                  "la estrella (luz propia de un planeta muy caliente). No es un "
                                  "planeta nuevo")
        else:
            s.secundario_de = p.numero
            if entero == 1:
                txt = f"mismo período que la señal #{p.numero}, desfasada {desfase:.2f} de órbita"
            else:
                txt = (f"período exactamente {razon:.3f}× el de la señal #{p.numero} y eventos en "
                       "una fase fija distinta de su órbita")
            relacion = (p.numero, txt + ": es su eclipse secundario (binaria eclipsante)")
    if relacion:
        s.pruebas["relacionada_con"] = relacion[0]

    # =========================== VEREDICTO ======================================
    if s.n_transitos == 0:
        s.veredicto = "DATOS INSUFICIENTES"
        motivos.append("ningún tránsito con datos suficientes alrededor: la señal cae en huecos o "
                       "bordes de los datos (típico de ruido instrumental)")
        s.motivos = motivos
        return s
    if s.sde < UMBRAL_SDE or s.snr < UMBRAL_SNR:
        s.veredicto = "SIN DETECCIÓN"
        motivos.insert(0, f"SDE={s.sde:.1f} (mín {UMBRAL_SDE:g}) y SNR={s.snr:.1f} "
                          f"(mín {UMBRAL_SNR:g}): no se distingue del ruido")
        s.motivos = motivos
        return s
    if n_nulos < 20:
        s.veredicto = "DATOS INSUFICIENTES"
        motivos.append(f"solo {n_nulos} zonas libres para medir el ruido (mín 20): "
                       "no se puede saber si la señal es real")
        s.motivos = motivos
        return s
    if s.n_transitos < MIN_TRANSITOS:
        s.veredicto = "DATOS INSUFICIENTES"
        motivos.append(f"solo {s.n_transitos} tránsito(s) con datos; se exigen {MIN_TRANSITOS} (criterio "
                       "de Kepler: con 2 eventos cualquier par de ruidos define un período). "
                       "Reanalizar con más sectores (--sectores 0)")
        s.motivos = motivos
        return s
    if s.pruebas.get("transitos_cerca_borde", 0) and (
            s.pruebas.get("snr_sin_bordes", 0) < UMBRAL_SNR or s.pruebas.get("transitos_lejos_borde", 0) < 2):
        s.veredicto = "DATOS INSUFICIENTES"
        motivos.append(f"{s.pruebas['transitos_cerca_borde']} tránsito(s) caen a menos de 12 h de un hueco "
                       f"de datos y sin ellos la señal no alcanza (SNR {s.pruebas.get('snr_sin_bordes', 0)}). "
                       "Típico de efectos instrumentales de TESS; hacen falta más sectores")
        s.motivos = motivos
        return s

    fp = []
    advertencias = []
    pi = s.pruebas.get("par_impar_sigma", 0)
    if pi > UMBRAL_FP_PAR_IMPAR:
        fp.append(f"tránsitos pares e impares de distinta profundidad "
                  f"({pi}σ): binaria eclipsante con período {2*P:.3f} d")
    elif pi > UMBRAL_SIGMA_TEST:
        advertencias.append(f"ADVERTENCIA: pares e impares difieren algo ({pi}σ); podría ser una "
                            f"binaria con período {2*P:.3f} d")
    sec_sigma = s.pruebas.get("secundario_sigma", 0)
    if UMBRAL_SIGMA_TEST < sec_sigma <= UMBRAL_FP_SECUNDARIO:
        advertencias.append(f"ADVERTENCIA: posible caída leve en la mitad de la órbita "
                            f"({s.pruebas.get('secundario_ppm', 0):.0f} ppm, {sec_sigma}σ); puede ser ruido "
                            "o el eclipse secundario de una binaria")
    if sec_sigma > UMBRAL_FP_SECUNDARIO:
        d2 = s.pruebas["secundario_ppm"]
        d1 = s.pruebas.get("profundidad_local_ppm", s.profundidad_ppm)
        en_v = s.pruebas.get("fraccion_plana", 1) <= UMBRAL_FORMA_V
        if d2 < 1500 and d2 < 0.1 * d1 and not en_v:
            # Un Júpiter muy caliente (ej. WASP-18 b) emite luz propia y su
            # eclipse detrás de la estrella se ve: hasta ~1000 ppm y siempre
            # mucho menor que el tránsito. No basta para descartar.
            advertencias.append(f"ADVERTENCIA: eclipse secundario chico ({d2:.0f} ppm, "
                                f"{s.pruebas['secundario_sigma']}σ). Compatible con un planeta muy "
                                "caliente, pero también con una binaria de compañera pequeña")
        else:
            fp.append(f"hay un eclipse secundario en fase 0.5 ({s.pruebas['secundario_sigma']}σ, "
                      f"{d2:.0f} ppm)" + (" y el tránsito tiene forma de V: binaria rasante" if en_v
                                          else ": demasiado brillante para venir de un planeta"))
    # Duración demasiado CORTA: solo posible si el planeta roza el borde de la estrella
    rel = s.duracion_h / max(dur_max_h, 1e-9)
    s.pruebas["duracion_relativa"] = round(float(rel), 2)
    if rel < (0.15 if estrella.supuesta else 0.25):
        if s.pruebas.get("fraccion_plana", 1) <= UMBRAL_FORMA_V:
            advertencias.append(f"ADVERTENCIA: dura {s.duracion_h:.1f} h, muy poco para esa órbita "
                                f"(un tránsito central duraría ~{dur_max_h:.1f} h). Con su forma de V puede "
                                "ser un planeta que roza el borde de la estrella, o una binaria rasante")
        else:
            # En la calibración con datos reales invertidos, este fue el canal
            # principal de falsas alarmas (destellos y picos instrumentales).
            fp.append(f"dura {s.duracion_h:.1f} h cuando un tránsito central duraría ~{dur_max_h:.1f} h, "
                      "y tiene forma de caja (un tránsito rasante tendría forma de V): típico de "
                      "destellos de la estrella o picos instrumentales")
    if s.radio_tierras > 25:
        fp.append(f"radio {s.radio_tierras:.0f} R⊕ (> 2.2 Júpiter): demasiado grande para un planeta")
    # Reglas que NO dependen de conocer la estrella:
    k = np.sqrt(max(s.profundidad_central_ppm, 0) / 1e6 * FACTOR_LIMBO)
    if k > 0.4:
        fp.append(f"el objeto taparía ~{k:.0%} del diámetro de la estrella: ningún planeta "
                  "conocido es tan grande en proporción (típico de dos estrellas)")
    elif (s.pruebas.get("fraccion_plana", 1) <= UMBRAL_FORMA_V
          and s.profundidad_central_ppm > 50_000):
        fp.append(f"forma de V y {s.profundidad_central_ppm/1e4:.0f} % de profundidad: un planeta "
                  "rozando el borde de la estrella no puede tapar tanto (binaria rasante)")
    if s.duracion_h > margen * dur_max_h:
        fp.append(f"dura {s.duracion_h:.1f} h, más de lo físicamente posible "
                  f"(~{dur_max_h:.1f} h) para esa órbita y esa estrella")
    if s.pruebas.get("transitos_faltantes", 0) > 0:
        fp.append(f"{s.pruebas['transitos_faltantes']} tránsito(s) con datos no muestran "
                  "la caída esperada: el período real probablemente es otro (alias)")
    elif s.pruebas.get("consistencia_z", 0) > 10:
        fp.append("los tránsitos individuales difieren muchísimo entre sí: "
                  "posible ruido instrumental o binaria")
    elif s.pruebas.get("consistencia_z", 0) > 5:
        advertencias.append("ADVERTENCIA: los tránsitos individuales varían más de lo esperado "
                            f"(z={s.pruebas['consistencia_z']}); revisar uno por uno")
    if s.pruebas.get("onda_menos_caja", -1) > 0:
        fp.append("la señal se ajusta mejor a una onda que a un tránsito: variabilidad de la estrella")
    if relacion:
        fp.append(relacion[1])

    if fp:
        s.veredicto = "PROBABLE FALSO POSITIVO"
        motivos += fp
    else:
        s.veredicto = "CANDIDATO PLANETARIO"
        # Forma en V: típica de binarias rasantes, pero también de planetas que
        # pasan por el borde de la estrella -> solo advertencia, no descarte.
        motivos += advertencias
        if s.snr < SNR_FUERTE:
            motivos.append(f"ADVERTENCIA: señal débil (SNR {s.snr:.1f} < {SNR_FUERTE:g}). En datos reales "
                           "de TESS muchas señales de este nivel resultan ser ruido; confirmar con más sectores")
        if len(tab2) == 0:
            motivos.append(f"ADVERTENCIA: período ambiguo. No hay datos en la mitad de la órbita, así "
                           f"que el período real también podría ser {P/2:.4f} d (los tránsitos "
                           "intermedios caerían en huecos de los datos)")
        epocas = tab[:, 0].astype(int)
        multiplos = [k for k in (2, 3) if len(set(epocas % k)) == 1]
        if multiplos:
            alternativas = " o ".join(f"{k * P:.4f} d" for k in multiplos)
            motivos.append(f"ADVERTENCIA: período ambiguo. Las épocas sin tránsito caen en huecos de "
                           f"los datos, así que el período real también podría ser {alternativas}")
        if s.pruebas.get("fraccion_plana", 1) <= UMBRAL_FORMA_V and s.snr > 20:
            motivos.append(f"ADVERTENCIA: tránsito con forma de V (fondo plano "
                           f"{s.pruebas['fraccion_plana']:.0%} de la duración). Puede ser una binaria "
                           "rasante, o un planeta que pasa rozando el borde de la estrella")
        motivos.append("pasa todas las pruebas automáticas. Aun así falta descartar "
                       "estrellas vecinas contaminando la luz y confirmar con otro método")
    s.motivos = motivos
    return s


# ---------------------------------------------------------------------------
# 5. FLUJO COMPLETO
# ---------------------------------------------------------------------------
def a_10min(lc):
    """Agrupa a 10 min si la cadencia es más fina (más rápido, mismo resultado
    para tránsitos de más de ~1 hora)."""
    if cadencia_dias(lc.time.value) < 9 / 1440:
        lc = lc.bin(time_bin_size=10 / 1440)
    return lc.remove_nans()


def fraccion_plana(t, f, periodo, t0, dur):
    """Ajusta un trapecio al tránsito plegado y devuelve qué fracción de su
    duración es fondo plano: ~0.7-0.95 para un planeta típico (forma de U),
    ~0 para una binaria rasante (forma de V). Un planeta que pasa rozando el
    borde de la estrella también da ~0: por eso es solo una advertencia."""
    x = fase_centrada(t, periodo, t0)
    sel = np.abs(x) < 1.2 * dur
    x, y = x[sel], f[sel] - 1
    if x.size < 20:
        return np.nan
    mejor = (np.inf, np.nan)
    for T in dur * np.linspace(0.7, 1.5, 17):
        for q in np.linspace(0, 1, 21):                      # q = fondo plano / total
            a = np.abs(x)
            borde = 0.5 * T * (1 - q)
            forma = np.clip((0.5 * T - a) / max(borde, 1e-9), 0, 1) if q < 1 else (a < 0.5 * T) * 1.0
            den = np.sum(forma ** 2)
            if den == 0:
                continue
            prof = -np.sum(forma * y) / den
            chi2 = np.sum((y + prof * forma) ** 2)
            if chi2 < mejor[0]:
                mejor = (chi2, q)
    return float(mejor[1])


def medir_senal(lc, ventana_dias, senales, estrella, n, P_inicial, t, f, dy):
    """Afina una señal a partir de un período aproximado y la mide.
    Devuelve (Senal sin veredicto, bls, plano_protegido, tab, tab2, err1, n_nulos)."""
    _, P, t0, dur = refinar(t, f, dy, P_inicial, estrella, t.max() - t.min())
    # Dos vueltas: limpiar protegiendo el tránsito (ventana >= 3 duraciones,
    # así no se lo "come") y volver a afinar con los datos ya protegidos.
    for _vuelta in range(2):
        ventana = max(ventana_dias, 3 * dur)
        prov = Senal(n, P, t0, dur * 24 * 1.3, 0, 0, 0, 0, 0, 0)
        plano_n = aplanar(lc, ventana, mascara=mascara_transitos(lc.time.value, senales + [prov]))
        pn = a_10min(plano_n[~mascara_transitos(plano_n.time.value, senales)] if senales else plano_n)
        t, f = pn.time.value, pn.flux.value
        dy = np.full_like(f, ruido_robusto(f[np.abs(fase_centrada(t, P, t0)) > dur]))
        bls_f, P, t0, dur = refinar(t, f, dy, P, estrella, t.max() - t.min())
    ventana_dias = max(ventana_dias, 3 * dur)

    fase = fase_centrada(t, P, t0)
    dentro, centro = np.abs(fase) < 0.5 * dur, np.abs(fase) < 0.25 * dur
    prof = 1 - np.mean(f[dentro]) if dentro.any() else 0.0
    prof_c = 1 - np.median(f[centro]) if centro.sum() >= 3 else prof
    # Radio: la profundidad en el centro es algo MAYOR que (Rp/R*)^2 porque el
    # centro de la estrella brilla más que el borde (oscurecimiento al limbo).
    radio = np.sqrt(max(prof_c, 0) * FACTOR_LIMBO) * estrella.radio_sol * R_SOL_EN_TIERRAS
    s = Senal(n, P, t0, dur * 24, prof * 1e6, prof_c * 1e6, 0.0, 0.0, 0, float(radio))
    s.pruebas["fraccion_plana"] = round(fraccion_plana(t, f, P, t0, dur), 2)
    tab, tab2, err1, n_nulos = medir_con_nulo(lc, ventana_dias, senales, P, t0, dur, semilla=n)
    return s, bls_f, plano_n, tab, tab2, err1, n_nulos


def es_alias_doble(tab, tab2, err1):
    """True si en fase 0.5 hay una caída IGUAL a la principal: no es una
    binaria, es el mismo planeta y el período real es la mitad."""
    if len(tab) == 0 or len(tab2) == 0:
        return False
    d1, e1 = media_ponderada(tab[:, 1], err1 / np.sqrt(tab[:, 2]))
    d2, e2 = media_ponderada(tab2[:, 1], err1 / np.sqrt(tab2[:, 2]))
    return (d2 / e2 > UMBRAL_SIGMA_TEST and d2 > 0.5 * d1
            and abs(d1 - d2) < UMBRAL_SIGMA_TEST * np.sqrt(e1 ** 2 + e2 ** 2 + (PISO_SISTEMATICO * d1) ** 2))


def analizar(lc, estrella: Estrella, max_senales=3, periodo_min=0.5,
             periodo_max=None, ventana_dias=None, verbose=True):
    """Busca hasta max_senales señales, una tras otra (al encontrar una, sus
    tránsitos se tapan y se busca de nuevo: así aparecen los otros planetas)."""
    lc = a_arreglo_simple(lc).remove_nans().normalize()
    avisos = []
    rango = float(np.subtract(*np.nanpercentile(lc.flux.value, [99, 1])))
    if rango > VARIABILIDAD_EXTREMA:
        avisos.append(f"ADVERTENCIA: la estrella varía {rango:.0%} en brillo (eclipses profundos o "
                      "variabilidad extrema); en estos casos el método es poco confiable")
        if verbose:
            print(avisos[-1])
    lc, _ = quitar_rotacion(lc, verbose=verbose)
    lc = a_10min(lc)     # todo el análisis a <=10 min: igual para tránsitos de >30 min y mucho más rápido
    if ventana_dias is None:
        ventana_dias, ruidos = elegir_ventana(lc)
        if verbose:
            print(f"Ventana de limpieza elegida: {ventana_dias} d (ruido en 2 h por ventana, ppm: {ruidos})")
            if ventana_dias < 0.75:
                print("  Aviso: estrella variable; tránsitos de más de "
                      f"~{ventana_dias * 24 / 3:.0f} h pueden quedar atenuados en la búsqueda.")
    t_all = lc.time.value
    bordes = bordes_de_datos(t_all)
    baseline = float(t_all.max() - t_all.min())
    if periodo_max is None:
        periodo_max = 0.75 * baseline      # pide 2 tránsitos; >baseline/2 depende de la fase
    periodo_max = min(periodo_max, 0.95 * baseline)

    senales: list[Senal] = []
    periodograma = None
    plano = aplanar(lc, ventana_dias)
    for n in range(1, max_senales + 1):
        if senales:   # re-limpiar protegiendo los tránsitos ya encontrados
            plano = aplanar(lc, ventana_dias, mascara=mascara_transitos(lc.time.value, senales))

        busq = a_10min(plano[~mascara_transitos(plano.time.value, senales)] if senales else plano)
        t, f = busq.time.value, busq.flux.value
        dy = np.full_like(f, ruido_robusto(f))
        bls, res, sde, factor = bls_periodograma(t, f, dy, periodo_min, periodo_max, estrella)
        if factor > 1 and verbose and n == 1:
            print(f"Aviso: grilla de períodos reducida {factor:.1f}x por la larga duración de "
                  "los datos; señales débiles de período largo pueden perderse.")
        i = int(np.argmax(sde))
        # Una señal fuerte "levanta" el piso del periodograma cerca de su
        # período real y le baja el SDE: entonces el máximo de SDE puede caer
        # en 2P o 3P. Se corrige mirando la potencia REAL en P/2 y P/3.
        for _ in range(3):
            mejor = None
            for k in (2, 3):
                cerca = np.abs(res.period / (res.period[i] / k) - 1) < 0.003
                if cerca.any():
                    j = np.flatnonzero(cerca)[np.argmax(res.power[cerca])]
                    # Margen de 20 %: si el período real es P/k, su potencia es ~k veces
                    # mayor; si son casi iguales, es ambiguo y se mantiene el original.
                    if res.power[j] > 1.2 * res.power[i] and (mejor is None or res.power[j] > res.power[mejor]):
                        mejor = j
            if mejor is None:
                break
            i = int(mejor)
        if periodograma is None:
            periodograma = (res.period, sde)

        try:
            s, bls_f, plano_n, tab, tab2, err1, n_nulos = medir_senal(
                lc, ventana_dias, senales, estrella, n, float(res.period[i]), t, f, dy)
            nota_alias = None
            if es_alias_doble(tab, tab2, err1) and s.periodo / 2 >= periodo_min:
                P_doble = s.periodo
                s, bls_f, plano_n, tab, tab2, err1, n_nulos = medir_senal(
                    lc, ventana_dias, senales, estrella, n, P_doble / 2, t, f, dy)
                nota_alias = (f"período corregido de {P_doble:.4f} a {s.periodo:.4f} d: en la mitad "
                              "de la órbita había una caída idéntica (era el mismo planeta)")
        except ValueError:
            s = Senal(n, float(res.period[i]), 0, 0, 0, 0, float(sde[i]), 0, 0, 0,
                      veredicto="SIN DETECCIÓN",
                      motivos=["el pico del periodograma no corresponde a ningún tránsito medible"])
            senales.append(s)
            break

        s.sde = float(sde[i])
        evaluar(s, tab, tab2, err1, n_nulos, bls_f, estrella, senales, bordes)
        if s.veredicto == "CANDIDATO PLANETARIO" and avisos:
            s.motivos = avisos + s.motivos
        if nota_alias:
            s.motivos.insert(0, nota_alias)
        if s.secundario_de and s.snr >= UMBRAL_SNR:   # corregir hacia atrás: era una binaria
            p = senales[s.secundario_de - 1]
            if p.veredicto == "CANDIDATO PLANETARIO":
                p.veredicto = "PROBABLE FALSO POSITIVO"
                p.motivos = [m for m in p.motivos if not m.startswith("pasa todas")]
                p.motivos.append(f"la señal #{s.numero} es su otro eclipse (misma órbita, otra "
                                 "fase): binaria eclipsante")
        senales.append(s)
        if s.veredicto == "SIN DETECCIÓN":
            break        # lo más fuerte que queda ya es ruido: parar
        plano = plano_n
    return senales, plano, periodograma


# ---------------------------------------------------------------------------
# 6. SIMULADOR (para practicar y para las pruebas de validación)
# ---------------------------------------------------------------------------
def modelo_transito(t, periodo, t0, prof, dur_h, u1=0.40, u2=0.25):
    """Tránsito realista (forma de U con oscurecimiento al limbo, entrada y
    salida graduales). No es una caja: probar BLS solo con cajas lo favorece."""
    dur = dur_h / 24
    norma = 1 - u1 / 3 - u2 / 6
    k = np.sqrt(prof * norma)                     # radio planeta / radio estrella
    d = np.abs(fase_centrada(t, periodo, t0)) / (dur / 2) * (1 + k)
    cubierto = np.clip((1 + k - d) / (2 * k), 0, 1)
    mu = np.sqrt(np.clip(1 - np.minimum(d, 1) ** 2, 0, 1))
    intensidad = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    return 1 - k ** 2 * intensidad / norma * cubierto


def simular(tipo="planeta", semilla=0, dias=27.0, cadencia_min=2.0, ruido_ppm=1000,
            planetas=None, variabilidad=0.002):
    """Curva de luz falsa estilo TESS: hueco a mitad del sector, ruido blanco +
    ruido rojo, variabilidad estelar. Devuelve (LightCurve, verdad)."""
    rng = np.random.default_rng(semilla)
    t = np.arange(0, dias, cadencia_min / 1440)
    mitad = dias / 2
    t = t[(t < mitad - 0.5) | (t > mitad + 0.5)]            # hueco de bajada de datos
    f = np.ones_like(t)
    f += variabilidad * np.sin(2 * np.pi * t / rng.uniform(3, 12) + rng.uniform(0, 6))
    blanco = rng.normal(0, ruido_ppm * 1e-6, t.size)
    rojo = np.convolve(rng.normal(0, 1, t.size), np.ones(30) / 30, mode="same")
    rojo *= 0.3 * ruido_ppm * 1e-6 / rojo.std()               # ruido correlacionado ~1 h
    f += blanco + rojo

    if planetas is None:
        planetas = {
            "planeta": [(3.7, 1.2, 0.004, 2.8)],
            "multi": [(3.3, 0.5, 0.004, 2.5), (7.7, 2.0, 0.0025, 3.5), (12.1, 4.0, 0.002, 4.5)],
            "ruido": [],
            "binaria": [],
        }[tipo]
    for P, t0, prof, dur_h in planetas:
        f *= modelo_transito(t, P, t0, prof, dur_h)
    if tipo == "binaria":   # eclipse primario 2 %, secundario 0.6 %, período 5.1 d
        f *= modelo_transito(t, 5.1, 1.0, 0.02, 4.0)
        f *= modelo_transito(t, 5.1, 1.0 + 2.55, 0.006, 4.0)
        planetas = [(5.1, 1.0, 0.02, 4.0)]
    lc = lk.LightCurve(time=t, flux=f, flux_err=np.full_like(t, ruido_ppm * 1e-6))
    return lc, planetas


# ---------------------------------------------------------------------------
# 7. INFORME Y GRÁFICO
# ---------------------------------------------------------------------------
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e6e5e0"


def aplicar_clasificador(senales, estrella, ruta):
    """Agrega a cada señal medible la probabilidad de ser planeta según el
    modelo entrenado con los TOIs que la NASA ya resolvió (ver ml/).
    Usa el modelo COMPLETO si se conocen radio, temperatura y gravedad de la
    estrella; si no, el BÁSICO (más débil: no conoce el tamaño del planeta).
    Es información EXTRA: no cambia el veredicto de las pruebas físicas."""
    import os
    import sys
    import joblib
    sys.path.insert(0, os.path.dirname(os.path.abspath(ruta)))
    from caracteristicas import desde_senal
    paquete = joblib.load(ruta)
    completa = (not estrella.supuesta and np.isfinite(estrella.teff) and np.isfinite(estrella.logg))
    usar = paquete["completo"] if completa else paquete["basico"]
    for s in senales:
        if s.veredicto in ("SIN DETECCIÓN", "DATOS INSUFICIENTES"):
            continue
        X = desde_senal(s, estrella, estrella.teff, estrella.logg)[usar["columnas"]]
        s.prob_planeta = float(usar["calibrador"].predict(usar["modelo"].predict_proba(X)[:, 1])[0])
        s.pruebas["modelo_ia"] = "completo" if completa else "basico"
        if s.veredicto == "CANDIDATO PLANETARIO" and s.prob_planeta < 0.2:
            s.motivos.insert(0, f"ADVERTENCIA: el modelo de IA le da solo {s.prob_planeta:.0%} de "
                                "probabilidad: se parece a falsos positivos conocidos, o es un tipo "
                                "de planeta poco común (el modelo castiga lo raro)")
    return senales


def imprimir(senales, estrella, nombre):
    print(f"\n================  RESULTADO: {nombre}  ================")
    if estrella.supuesta:
        print("Estrella: sin datos de radio/masa -> se asume una estrella como el Sol.")
    else:
        print(f"Estrella: radio {estrella.radio_sol:.2f} R☉, masa {estrella.masa_sol:.2f} M☉ (catálogo TIC)")
    for s in senales:
        print(f"\n--- Señal #{s.numero}: {s.veredicto} ---")
        if s.veredicto != "SIN DETECCIÓN":
            print(f"  Período      {s.periodo:.5f} d      Duración {s.duracion_h:.2f} h")
            print(f"  Profundidad  {s.profundidad_central_ppm:.0f} ppm ({s.profundidad_central_ppm/1e4:.3f} %)"
                  f"   Radio ≈ {s.radio_tierras:.2f} R⊕ ({s.radio_tierras/R_JUP_EN_TIERRAS:.2f} R♃)")
            print(f"  Tránsitos    {s.n_transitos}          SDE {s.sde:.1f}   SNR {s.snr:.1f}")
            if np.isfinite(s.prob_planeta):
                basico = s.pruebas.get("modelo_ia") == "basico"
                print(f"  Modelo IA    {s.prob_planeta:.0%} de probabilidad de ser planeta"
                      + ("  (modelo básico, sin datos de la estrella: bastante menos confiable)"
                         if basico else ""))
        for m in s.motivos:
            print(f"  · {m}")
    print("\nPruebas (números crudos, para revisar):")
    for s in senales:
        limpio = {k: (round(float(v), 2) if isinstance(v, (float, np.floating))
                      else int(v) if isinstance(v, (int, np.integer)) else v)
                  for k, v in s.pruebas.items()}
        print(f"  #{s.numero}: {limpio}")
    print("\nLímites del método (no se pueden eliminar, solo tener en cuenta):")
    print("  · Solo se ven planetas cuya órbita pasa justo frente a la estrella (de canto).")
    print("  · Favorece planetas grandes y de período corto; los chicos y lejanos se pierden.")
    print("  · Se exigen 3 tránsitos: no se confirman períodos mayores a ~1/2 del tiempo observado.")
    print("  · Una binaria eclipsante de FONDO (otra estrella en el mismo píxel) puede imitar")
    print("    a un planeta perfecto. Eso no se descarta con la curva de luz sola.")
    if estrella.supuesta:
        print("  · Sin radio/masa de la estrella, el radio del planeta y la prueba de duración")
        print("    son aproximados (binarias de estrellas gemelas pueden pasar como candidatos).")


def graficar(lc_plano, senales, periodograma, nombre, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA2,
                         "xtick.color": TINTA2, "ytick.color": TINTA2,
                         "axes.spines.top": False, "axes.spines.right": False})
    reales = [s for s in senales if s.veredicto != "SIN DETECCIÓN"]
    filas = 2 + len(reales)
    fig = plt.figure(figsize=(10, 3 + 2.6 * filas))
    gs = GridSpec(filas, 2, figure=fig, hspace=0.6, wspace=0.25)
    t, f = lc_plano.time.value, lc_plano.flux.value

    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, f, ".", ms=1, color="#9b9a95", rasterized=True)
    for j, s in enumerate(reales):
        tt = s.t0 + s.periodo * np.arange(np.ceil((t.min() - s.t0) / s.periodo),
                                          np.floor((t.max() - s.t0) / s.periodo) + 1)
        ax.plot(tt, np.full_like(tt, np.nanpercentile(f, 0.2)), "^", ms=7,
                color=COLORES[j % 5], label=f"#{s.numero} · {s.veredicto.lower()}")
    ax.set(title=f"Curva de luz limpia · {nombre}", xlabel="Tiempo (días)", ylabel="Brillo relativo")
    if reales:
        ax.legend(loc="upper right", frameon=True, facecolor="#fcfcfb", edgecolor="none",
                  framealpha=0.9, fontsize=8, ncol=min(len(reales), 2))

    ax = fig.add_subplot(gs[1, :])
    per, sde = periodograma
    ax.plot(per, sde, lw=0.8, color=COLORES[0])
    ax.axhline(UMBRAL_SDE, color=TINTA2, ls="--", lw=1)
    ax.text(per.max(), UMBRAL_SDE, f" umbral SDE={UMBRAL_SDE:g}", va="bottom", ha="right", color=TINTA2)
    ax.set(xscale="log", title="Periodograma BLS de la 1ª búsqueda (picos sobre la línea = significativos)",
           xlabel="Período (días, escala log)", ylabel="SDE")

    for j, s in enumerate(reales):
        c = COLORES[j % 5]
        otras = mascara_transitos(t, [o for o in reales if o is not s])
        tt, ff = t[~otras], f[~otras]
        horas = fase_centrada(tt, s.periodo, s.t0) * 24
        ventana = max(3 * s.duracion_h, 6)
        epoca = np.round((tt - s.t0) / s.periodo).astype(int)

        ax = fig.add_subplot(gs[2 + j, 0])
        sel = np.abs(horas) < ventana
        ax.plot(horas[sel], ff[sel], ".", ms=1.5, color="#c3c2b7", rasterized=True)
        bordes = np.linspace(-ventana, ventana, 61)
        for paridad, estilo, etiqueta in ((0, "-", "pares"), (1, "--", "impares")):
            m = sel & (epoca % 2 == paridad)
            idx = np.digitize(horas[m], bordes)
            med = [np.mean(ff[m][idx == b]) if np.sum(idx == b) > 2 else np.nan
                   for b in range(1, len(bordes))]
            ax.plot(0.5 * (bordes[1:] + bordes[:-1]), med, estilo, lw=2, color=c, label=etiqueta)
        ax.set(title=f"#{s.numero}: P={s.periodo:.4f} d · {s.radio_tierras:.1f} R⊕ · {s.veredicto}",
               xlabel="Horas desde el centro del tránsito", ylabel="Brillo")
        ax.legend(frameon=False, fontsize=8, loc="lower left")

        ax = fig.add_subplot(gs[2 + j, 1], sharey=fig.axes[-1])
        h2 = fase_centrada(tt, s.periodo, s.t0 + s.periodo / 2) * 24
        sel = np.abs(h2) < ventana
        ax.plot(h2[sel], ff[sel], ".", ms=1.5, color="#c3c2b7", rasterized=True)
        idx = np.digitize(h2[sel], bordes)
        med = [np.mean(ff[sel][idx == b]) if np.sum(idx == b) > 2 else np.nan
               for b in range(1, len(bordes))]
        ax.plot(0.5 * (bordes[1:] + bordes[:-1]), med, lw=2, color=c)
        ax.set(title="Mitad de la órbita (¿eclipse secundario?)", xlabel="Horas desde fase 0.5")

    fig.savefig(salida, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"\nGráfico guardado en {salida}")


def main():
    p = argparse.ArgumentParser(description="Buscador de tránsitos planetarios con validación")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--estrella", help='Nombre o ID: "Pi Mensae", "TOI 270", "TIC 261136679"')
    g.add_argument("--demo", choices=["planeta", "multi", "binaria", "ruido"], nargs="?",
                   const="planeta", help="Estrella simulada (sabemos la respuesta correcta)")
    p.add_argument("--sectores", type=int, default=1, help="Cuántos sectores TESS usar (0 = todos)")
    p.add_argument("--autor", default=None, help="SPOC, TESS-SPOC o QLP (por defecto: el mejor disponible)")
    p.add_argument("--max-senales", type=int, default=3)
    p.add_argument("--periodo-min", type=float, default=0.5)
    p.add_argument("--periodo-max", type=float, default=None)
    p.add_argument("--ventana", type=float, default=None,
                   help="Ventana de limpieza en días (por defecto: automática)")
    p.add_argument("--salida", default="resultado.png")
    p.add_argument("--json", default=None, help="Guardar resultado en JSON")
    p.add_argument("--clasificador", default=None,
                   help="Ruta a clasificador_toi.joblib para agregar la probabilidad del modelo de IA")
    a = p.parse_args()

    if a.estrella:
        lc, _ = descargar(a.estrella, a.sectores, a.autor)
        estrella, nombre = parametros_estrella(lc), a.estrella
    else:
        tipo = a.demo or "planeta"
        lc, verdad = simular(tipo, semilla=7)
        estrella, nombre = Estrella(), f"DEMO {tipo}"
        print("Respuesta correcta (P, t0, profundidad, duración h):", verdad or "nada, solo ruido")

    senales, plano, periodograma = analizar(lc, estrella, a.max_senales, a.periodo_min,
                                            a.periodo_max, a.ventana)
    if a.clasificador:
        aplicar_clasificador(senales, estrella, a.clasificador)
    imprimir(senales, estrella, nombre)
    graficar(plano, senales, periodograma, nombre, a.salida)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump([asdict(s) for s in senales], fh, ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
