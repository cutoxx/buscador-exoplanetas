"""Prueba el clasificador con estrellas reales EXCLUIDAS del entrenamiento.
Compara la probabilidad con las mediciones del buscador vs con las del catálogo."""
import os
import sys
import warnings

import joblib
import pandas as pd

warnings.filterwarnings("ignore")
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
import buscador_transitos as bt  # noqa: E402
from caracteristicas import desde_catalogo  # noqa: E402

RUTA = os.path.join(AQUI, "clasificador_toi.joblib")
paq = joblib.load(RUTA)
cat = pd.read_csv(os.path.join(AQUI, "datos", "toi_exofop.csv"))

for nombre, tic, sectores in [("Pi Mensae", 261136679, 1), ("TOI-270", 259377017, 3),
                              ("WASP-18", 100100827, 1), ("CM Draconis", 199574208, 1)]:
    assert tic in paq["excluidas"], "¡esta estrella se usó para entrenar!"
    lc, _ = bt.descargar(f"TIC {tic}", sectores, None)
    est = bt.parametros_estrella(lc)
    senales, _, _ = bt.analizar(lc, est, max_senales=4, verbose=False)
    bt.aplicar_clasificador(senales, est, RUTA)
    print(f"\n### {nombre}  (Teff {est.teff:.0f} K, R {est.radio_sol:.2f} R☉)")
    for s in senales:
        if s.veredicto in ("SIN DETECCIÓN", "DATOS INSUFICIENTES"):
            continue
        print(f"  buscador: P={s.periodo:.4f} d  {s.veredicto:24s} IA={s.prob_planeta:.0%}")
    filas = cat[cat["TIC ID"] == tic]
    if len(filas):
        c = paq["completo"]
        p = c["calibrador"].predict(c["modelo"].predict_proba(desde_catalogo(filas)[c["columnas"]])[:, 1])
        for (_, f), pp in zip(filas.iterrows(), p):
            print(f"  catálogo: P={f['Period (days)']:.4f} d  TOI {f['TOI']} "
                  f"({f['TFOPWG Disposition']}) IA={pp:.0%}")
    sys.stdout.flush()
