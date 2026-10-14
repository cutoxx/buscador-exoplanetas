"""
entrenar_clasificador.py
------------------------
Entrena modelos que estiman la probabilidad de que un candidato a tránsito sea
un PLANETA REAL, aprendiendo de los candidatos de TESS (TOIs) que la NASA ya
resolvió: CP/KP = planeta (1), FP/FA = falso positivo (0).

Dos modelos:
  * completo: con datos de la estrella (radio, temperatura, gravedad).
  * básico:   solo período, duración y profundidad (estrellas sin datos).

Cuidados contra trampas y sesgos (ver README):
  1. Solo variables que el buscador también calcula (nada de comentarios,
     disposición del pipeline, número de observaciones de seguimiento, etc.).
  2. Sin SNR ni magnitud: en el catálogo el SNR sale de TODOS los sectores
     observados hasta hoy, no de una detección -> no es comparable.
  3. "Dato faltante" nunca es pista (por eso hay dos modelos).
  4. Pesos inversos a la probabilidad de recibir veredicto (las estrellas
     brillantes se revisan mucho más -> sesgo de selección).
  5. Validación agrupada POR ESTRELLA y validación TEMPORAL.
  6. Las estrellas usadas después para probar el buscador quedan fuera.

Uso:  python entrenar_clasificador.py  [--catalogo datos/toi_exofop.csv]
Descarga del catálogo: https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from caracteristicas import BASICAS, COLUMNAS, ESTELARES, desde_catalogo

EXCLUIR_TIC = [261136679, 259377017, 100100827, 199574208]   # estrellas de prueba
SEMILLA = 42


def cargar(ruta):
    d = pd.read_csv(ruta)
    d["fecha"] = pd.to_datetime(d["Date TOI Alerted (UTC)"])
    d["etiquetado"] = d["TFOPWG Disposition"].isin(["CP", "KP", "FP", "FA"])
    # Probabilidad de recibir veredicto según el brillo (para corregir el sesgo)
    bins = pd.cut(d["TESS Mag"], [-5, 8, 9, 10, 11, 12, 13, 14, 25])
    tasa = d.groupby(bins, observed=False)["etiquetado"].transform("mean")
    d["peso"] = 1 / np.clip(tasa, 0.05, 1)
    lab = d[d["etiquetado"] & ~d["TIC ID"].isin(EXCLUIR_TIC)].copy()
    lab["y"] = lab["TFOPWG Disposition"].isin(["CP", "KP"]).astype(int)
    return d, lab


def modelo():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=1.0, early_stopping=False, random_state=SEMILLA)


def metricas(y, p, w=None):
    return dict(roc_auc=round(roc_auc_score(y, p, sample_weight=w), 3),
                pr_auc=round(average_precision_score(y, p, sample_weight=w), 3),
                brier=round(brier_score_loss(y, p, sample_weight=w), 3),
                exactitud_0p5=round(float(np.average((p >= 0.5) == y, weights=w)), 3),
                n=int(len(y)))


def tabla_sesgo(lab, p, columna, bordes, nombre):
    filas = []
    for g, idx in lab.groupby(pd.cut(lab[columna], bordes), observed=True).groups.items():
        sel = lab.index.get_indexer(idx)
        y = lab["y"].values[sel]
        if len(set(y)) < 2 or len(y) < 30:
            continue
        filas.append(dict(variable=nombre, rango=str(g), n=len(y), frac_planetas=round(y.mean(), 2),
                          prob_media=round(p[sel].mean(), 2), roc_auc=round(roc_auc_score(y, p[sel]), 3)))
    return filas


def entrenar(lab, columnas, nombre):
    print(f"\n==================== MODELO {nombre.upper()} ====================")
    X = desde_catalogo(lab)[columnas]
    X.index = lab.index
    y, grupos = lab["y"], lab["TIC ID"]
    w = lab["peso"] / lab["peso"].mean()
    print(f"Candidatos: {len(lab)} ({y.sum()} planetas, {len(y)-y.sum()} falsos). Variables: {columnas}")
    inf = {"n": len(lab), "variables": columnas}

    # 1. Validación cruzada agrupada por estrella
    p_cv = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(X, y, grupos):
        p_cv[te] = modelo().fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    inf["cv_por_estrella"] = metricas(y, p_cv)
    inf["cv_ponderada"] = metricas(y, p_cv, w)
    sin_kp = (lab["TFOPWG Disposition"] != "KP").values
    inf["cv_sin_KP"] = metricas(y[sin_kp], p_cv[sin_kp])
    print("1) Validación cruzada por estrella:", inf["cv_por_estrella"])
    print("   ponderada por sesgo de selección:", inf["cv_ponderada"])
    print("   solo descubrimientos de TESS (sin KP):", inf["cv_sin_KP"])

    # 2. Validación temporal
    corte = lab["fecha"].quantile(0.7)
    tr, te = (lab["fecha"] < corte).values, (lab["fecha"] >= corte).values
    m_t = modelo().fit(X[tr], y[tr], sample_weight=w[tr])
    inf["temporal"] = dict(corte=str(corte.date()), **metricas(y[te], m_t.predict_proba(X[te])[:, 1]))
    print(f"2) Temporal (entrena antes de {corte.date()}, prueba después):", inf["temporal"])

    # 3. Sesgos por subgrupo
    lab = lab.assign(fuente=np.where(lab["Source"].astype(str).str.startswith("qlp"), 1.5, 0.5))
    sesgos = (tabla_sesgo(lab, p_cv, "TESS Mag", [-5, 9, 11, 13, 25], "magnitud TESS")
              + tabla_sesgo(lab, p_cv, "Period (days)", [0, 2, 5, 10, 1000], "período (d)")
              + tabla_sesgo(lab, p_cv, "Planet Radius (R_Earth)", [0, 4, 8, 15, 1000], "radio (R⊕)")
              + tabla_sesgo(lab, p_cv, "fuente", [0, 1, 2], "pipeline (0-1 SPOC, 1-2 QLP)"))
    inf["sesgos"] = sesgos
    print("3) Rendimiento por subgrupo:")
    print(pd.DataFrame(sesgos).to_string(index=False))

    # 4. Calibración
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(p_cv, y)
    p_cal = iso.predict(p_cv)
    cal = []
    for lo in np.arange(0, 1, 0.2):
        s = (p_cal >= lo) & (p_cal < lo + 0.2 + 1e-9)
        if s.sum() >= 20:
            cal.append(dict(prob_dicha=f"{lo:.1f}-{lo+0.2:.1f}", n=int(s.sum()),
                            prob_real=round(float(y[s].mean()), 2)))
    inf["calibracion"] = cal
    print("4) Calibración (lo que dice vs lo que pasa):")
    print(pd.DataFrame(cal).to_string(index=False))

    # 5. Importancia por permutación (fuera de muestra)
    imp = permutation_importance(m_t, X[te], y[te], scoring="roc_auc", n_repeats=10,
                                 random_state=SEMILLA)
    inf["importancia"] = {c: round(float(v), 3) for c, v in
                          sorted(zip(columnas, imp.importances_mean), key=lambda x: -x[1])}
    print("5) Importancia:", inf["importancia"])

    final = modelo().fit(X, y, sample_weight=w)
    return {"modelo": final, "calibrador": iso, "columnas": columnas}, inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogo", default="datos/toi_exofop.csv")
    a = ap.parse_args()
    todo, lab = cargar(a.catalogo)
    con_estrella = desde_catalogo(lab)[ESTELARES].notna().all(axis=1).values
    completo, inf_c = entrenar(lab[con_estrella], COLUMNAS, "completo (con datos de la estrella)")
    basico, inf_b = entrenar(lab, BASICAS, "básico (sin datos de la estrella)")

    # Candidatos sin veredicto (PC) puntuados con el modelo que corresponda
    pcs = todo[todo["TFOPWG Disposition"] == "PC"]
    Xp = desde_catalogo(pcs)
    usa_c = Xp[ESTELARES].notna().all(axis=1).values
    p = np.empty(len(pcs))
    p[usa_c] = completo["calibrador"].predict(completo["modelo"].predict_proba(Xp[usa_c][COLUMNAS])[:, 1])
    p[~usa_c] = basico["calibrador"].predict(basico["modelo"].predict_proba(Xp[~usa_c][BASICAS])[:, 1])
    pc = dict(n=len(pcs), mediana=round(float(np.median(p)), 2),
              frac_mayor_0p8=round(float((p > 0.8).mean()), 2), frac_menor_0p2=round(float((p < 0.2).mean()), 2))
    print("\nCandidatos SIN veredicto (PC) puntuados:", pc)

    informe = {"completo": inf_c, "basico": inf_b, "pc_sin_veredicto": pc,
               "catalogo": a.catalogo, "excluidas": EXCLUIR_TIC}
    joblib.dump({"completo": completo, "basico": basico, "excluidas": EXCLUIR_TIC,
                 "informe": informe}, "clasificador_toi.joblib")
    with open("informe_clasificador.json", "w") as fh:
        json.dump(informe, fh, ensure_ascii=False, indent=2, default=str)
    print("\nModelos guardados en clasificador_toi.joblib")


if __name__ == "__main__":
    main()
