"""
caracteristicas.py
------------------
Las MISMAS variables, calculadas igual, para el catálogo TOI (entrenamiento) y
para las señales del buscador (uso). Si se calcularan distinto, el modelo
vería en la práctica datos diferentes a los que aprendió.
"""
import numpy as np
import pandas as pd

G, R_SOL, M_SOL = 6.674e-8, 6.957e10, 1.989e33

COLUMNAS = ["log_periodo", "log_duracion_h", "log_profundidad_ppm", "log_radio_tierras",
            "radio_estrella", "teff", "logg", "duracion_relativa"]
# Modelo básico, para estrellas sin datos en el catálogo. NO se usa "dato
# faltante" como pista: en el catálogo, a los falsos positivos les faltan los
# datos de la estrella mucho más seguido (7 % planetas vs 55 %), y un modelo
# único aprendía "faltan datos -> falso positivo".
BASICAS = ["log_periodo", "log_duracion_h", "log_profundidad_ppm"]
ESTELARES = ["radio_estrella", "teff", "logg"]


def duracion_central_h(periodo, radio_estrella, masa_estrella, prof_ppm):
    """Duración (h) de un tránsito central según la 3ª ley de Kepler."""
    periodo = np.asarray(periodo, dtype=float)
    a = (G * masa_estrella * M_SOL * (periodo * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    k = np.sqrt(np.clip(np.asarray(prof_ppm, dtype=float), 0, None) / 1e6)
    arg = np.clip(radio_estrella * R_SOL * (1 + k) / a, 0, 1)
    return periodo * 24 / np.pi * np.arcsin(arg)


def masa_desde(radio, logg, masa):
    """Masa de la estrella: la del catálogo, o a partir de log g y radio."""
    m = np.asarray(masa, dtype=float)
    desde_g = (10 ** np.asarray(logg, dtype=float)) * (np.asarray(radio, dtype=float) * R_SOL) ** 2 / G / M_SOL
    return np.where(np.isfinite(m) & (m > 0), m, desde_g)


def _armar(P, dur_h, prof, rp, rs, teff, logg, masa):
    P, dur_h, prof, rp = (np.asarray(x, dtype=float) for x in (P, dur_h, prof, rp))
    rs, teff, logg = (np.asarray(x, dtype=float) for x in (rs, teff, logg))
    m = masa_desde(rs, logg, masa)
    with np.errstate(all="ignore"):
        t_esp = duracion_central_h(P, rs, m, prof)
        return pd.DataFrame({
            "log_periodo": np.log10(P),
            "log_duracion_h": np.log10(dur_h),
            "log_profundidad_ppm": np.log10(np.clip(prof, 1, None)),
            "log_radio_tierras": np.log10(np.clip(rp, 0.1, None)),
            "radio_estrella": rs,
            "teff": teff,
            "logg": logg,
            # >1: dura más de lo que permite la física para esa estrella (típico
            # de binarias o de una estrella mal caracterizada); <1: tránsito rasante.
            "duracion_relativa": dur_h / t_esp,
        })[COLUMNAS].replace([np.inf, -np.inf], np.nan)   # divisiones por cero -> dato faltante


def desde_catalogo(df):
    return _armar(df["Period (days)"], df["Duration (hours)"], df["Depth (ppm)"],
                  df["Planet Radius (R_Earth)"], df["Stellar Radius (R_Sun)"],
                  df["Stellar Eff Temp (K)"], df["Stellar log(g) (cm/s^2)"],
                  df["Stellar Mass (M_Sun)"])


def desde_senal(s, estrella, teff=np.nan, logg=np.nan):
    """s: Senal del buscador; estrella: Estrella del buscador."""
    rs = np.nan if estrella.supuesta else estrella.radio_sol
    masa = np.nan if estrella.supuesta else estrella.masa_sol
    rp = s.radio_tierras if not estrella.supuesta else np.nan
    return _armar([s.periodo], [s.duracion_h], [s.profundidad_central_ppm], [rp],
                  [rs], [teff], [logg], [masa])
