"""
validar_reales.py
-----------------
Corre buscador_transitos.py sobre estrellas REALES de TESS con respuesta conocida.

  Pi Mensae   (TIC 261136679): planeta c, P = 6.268 d, ~2 R⊕
  TOI-270     (TIC 259377017): 3 planetas, P = 3.360, 5.660, 11.380 d (sectores 3-5)
  WASP-18     (TIC 100100827): Júpiter caliente, P = 0.9415 d, con eclipse secundario real
  CM Draconis (TIC 199574208): binaria eclipsante de enanas M, P = 1.2684 d

Además, prueba nula con datos reales: se invierte el brillo (las caídas pasan
a ser subidas). Así se conserva el ruido real de TESS, pero ya no puede haber
tránsitos: cualquier "candidato" ahí sería una falsa alarma.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import buscador_transitos as bt  # noqa: E402

CASOS = [
    ("Pi Mensae", "TIC 261136679", 1, [6.2679], "planeta"),
    ("TOI-270", "TIC 259377017", 3, [3.3601, 5.6605, 11.380], "planeta"),
    ("WASP-18", "TIC 100100827", 1, [0.94145], "planeta"),
    ("CM Draconis", "TIC 199574208", 1, [1.26839], "binaria"),
]


def coincide(P, reales):
    for R in reales:
        for k in (1, 2, 0.5):
            if abs(P / (R * k) - 1) < 0.003:
                return R, k
    return None


def main(solo=None):
    for nombre, tic, sectores, periodos, tipo in CASOS:
        if solo and solo not in nombre:
            continue
        print(f"\n######## {nombre} ({tic}) — esperado: {tipo}, P = {periodos}")
        lc, autor = bt.descargar(tic, sectores, None)
        est = bt.parametros_estrella(lc)
        print(f"Estrella: R={est.radio_sol:.2f} R☉ M={est.masa_sol:.2f} M☉ (supuesta={est.supuesta})")
        senales, plano, pg = bt.analizar(lc, est, max_senales=4)
        for s in senales:
            c = coincide(s.periodo, periodos)
            marca = f"= {c[0]} d" + ("" if c[1] == 1 else f" (alias x{c[1]})") if c else "(no esperado)"
            print(f"  #{s.numero} P={s.periodo:.5f} {marca:22s} {s.veredicto:24s} "
                  f"R={s.radio_tierras:5.2f} R⊕ SDE={s.sde:5.1f} SNR={s.snr:6.1f}")
            for m in s.motivos:
                print(f"       · {m}")
        bt.graficar(plano, senales, pg, nombre, f"real_{nombre.replace(' ', '_')}.png")

        # Prueba nula: brillo invertido
        inv = lc.remove_nans().normalize()
        inv.flux = 2 - inv.flux
        nul, _, _ = bt.analizar(inv, est, max_senales=2, verbose=False)
        cands = [s for s in nul if s.veredicto == "CANDIDATO PLANETARIO"]
        print(f"  Prueba nula (brillo invertido): {len(cands)} candidatos "
              f"{'(OK)' if not cands else '(FALSA ALARMA)'}; "
              + ", ".join(f"P={s.periodo:.3f} {s.veredicto} SDE={s.sde:.1f} SNR={s.snr:.1f}" for s in nul))
        sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
