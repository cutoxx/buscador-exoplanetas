"""¿Ayuda el modelo donde las pruebas físicas no alcanzan? Planetas simulados vs
binarias gemelas rasantes (que pasan las pruebas con solo una advertencia).
Las simulaciones son de estrellas tipo Sol, así que se le dan esos datos."""
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import lightkurve as lk

warnings.filterwarnings("ignore")
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
import buscador_transitos as bt  # noqa: E402
import pruebas_validacion as pv  # noqa: E402

RUTA = os.path.join(AQUI, "clasificador_toi.joblib")
SOL = bt.Estrella(1.0, 1.0, False, 5778.0, 4.44)


def correr(tarea):
    tipo, sem = tarea
    if tipo == "planeta":
        rng = np.random.default_rng(2000 + sem)
        p = pv.parametros_ruido(rng, permitir_rapidos=False)
        p["var_amp"] = min(p["var_amp"], 5e-3)
        t, f = pv.curva_base(rng, **p)
        P = float(np.exp(rng.uniform(np.log(0.7), np.log(13))))
        b = float(rng.uniform(0, 0.85))
        prof = float(np.exp(rng.uniform(np.log(1000), np.log(8000)))) * 1e-6   # señales claras
        f = f * bt.modelo_transito(t, P, float(rng.uniform(0, P)), prof, float(pv.t14_sol(P, b, np.sqrt(prof))))
    else:  # gemelas rasantes: misma receta que pruebas_validacion
        rng = np.random.default_rng(5000 + sem)
        p = pv.parametros_ruido(rng, permitir_rapidos=False)
        p["var_amp"] = min(p["var_amp"], 3e-3)
        t, f = pv.curva_base(rng, **p)
        P = float(np.exp(rng.uniform(np.log(1.0), np.log(10))))
        t0 = float(rng.uniform(0, P))
        prof1 = float(np.exp(rng.uniform(np.log(0.003), np.log(0.03))))
        dur = float(pv.t14_sol(P, 0.3, 0.8) * rng.uniform(0.8, 1.2)) * 0.6
        f = f * pv.eclipse_v(t, P, t0, prof1, dur) * pv.eclipse_v(t, P, t0 + P / 2, prof1 * rng.uniform(0.9, 1), dur)
    senales, _, _ = bt.analizar(lk.LightCurve(time=t, flux=f), SOL, max_senales=1, verbose=False)
    bt.aplicar_clasificador(senales, SOL, RUTA)
    s = senales[0]
    return tipo, s.veredicto, s.prob_planeta


if __name__ == "__main__":
    tareas = [("planeta", i) for i in range(24)] + [("gemelas_rasantes", i) for i in range(24)]
    with Pool(2) as pool:
        res = pool.map(correr, tareas)
    for tipo in ("planeta", "gemelas_rasantes"):
        g = [(v, p) for t, v, p in res if t == tipo]
        cand = [p for v, p in g if v == "CANDIDATO PLANETARIO"]
        print(f"\n{tipo}: {len(g)} casos, {len(cand)} salen como 'candidato' de las pruebas físicas")
        if cand:
            c = np.array(cand)
            print(f"   prob. IA de esos candidatos: mediana {np.median(c):.0%}, "
                  f">=50 %: {np.mean(c >= 0.5):.0%}, <20 %: {np.mean(c < 0.2):.0%}")
            print("   valores:", " ".join(f"{x:.0%}" for x in sorted(c)))
