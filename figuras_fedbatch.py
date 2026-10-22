"""
Figuras del fed-batch (en inglés) y sensibilidad del óptimo, a partir de los resultados ya calculados.
No vuelve a correr el AG. Lee resultados_fedbatch.xlsx y resultados_NSGA3_por_fermentacion.xlsx
(misma carpeta) y vuelve a simular el modelo con las mismas ecuaciones, condiciones y tolerancias
que FedBatch_optimizacion_validacion.py. Al inicio comprueba que la simulación coincide con el Excel.

Salidas:
    fedbatch_optimo.png             biomasa, glucosa y O2 en el óptimo frente a S_in = 15.43 g/L, solo modelo (una columna)
    fedbatch_validacion.png         predicción con S_in = 15.43 g/L vs. medición (2 x 2 paneles; una columna)
    fedbatch_convergencia_AG.png    convergencia del AG, 10 semillas (una columna)
    sensibilidad_fedbatch.xlsx      S_in* y X(t_f) frente al límite de O2 (5-30 %) y con los parámetros
                                    de Chebyshev de F2

Uso:
    python figuras_fedbatch.py
"""
import os

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
XL_FB = os.path.join(AQUI, "resultados_fedbatch.xlsx")
XL_CAL = os.path.join(AQUI, "resultados_NSGA3_por_fermentacion.xlsx")

# Constantes y condiciones (hoja "configuracion" de resultados_fedbatch.xlsx)
O2S, QO2M = 0.005474, 0.0144
F, T_ON, V0, V_FIN, TF = 0.27, 5.5, 1.5, 2.0, 20.0
T_OFF = T_ON + (V_FIN - V0) / F
Y0 = [0.416667, 4.71, 0.004709, 0.005201, V0]
S_EXP = 15.43
RTOL, ATOL = 1e-9, 1e-12
MALLA = np.round(np.arange(T_ON, TF + 1e-9, 0.01), 2)      # malla de las restricciones

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 7.5,
                     "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.6})


def rhs(t, y, th, s_in, f):
    mumax, Ks, KO2, Yxs, Yxp, beta, KLa, YxO2 = th
    X, S, P, O, V = y
    Sp, Op = max(S, 0.0), max(O, 0.0)
    mu = mumax * Sp / (Ks + Sp) * Op / (KO2 + Op)
    D = f / V
    return [mu * X - D * X,
            D * (s_in - S) - mu * X / Yxs,
            (mu / Yxp + beta) * X - D * P,
            KLa * (O2S - O) - (mu / YxO2 + QO2M) * X - D * O,
            f]


def simular(th, s_in, t_eval):
    """Integración por tramos (antes, durante y después de la alimentación)."""
    y = np.array(Y0, float)
    T, Y = [], []
    for a, b, f in [(0.0, T_ON, 0.0), (T_ON, T_OFF, F), (T_OFF, TF, 0.0)]:
        te = t_eval[(t_eval >= a) & (t_eval <= b)]
        if len(te):
            s = solve_ivp(rhs, (a, b), y, method="LSODA", args=(th, s_in, f), t_eval=te, rtol=RTOL, atol=ATOL)
            T.append(s.t); Y.append(s.y)
        y = solve_ivp(rhs, (a, b), y, method="LSODA", args=(th, s_in, f), rtol=RTOL, atol=ATOL).y[:, -1]
    T = np.concatenate(T); Y = np.concatenate(Y, axis=1)
    _, idx = np.unique(T, return_index=True)
    return T[idx], Y[:, idx]


def o2_min_alimentacion(th, s_in):
    _, Y = simular(th, s_in, MALLA)
    return Y[3].min(), Y[0][-1]


def optimo_por_biseccion(th, frac, a=0.0, b=30.0, tol=1e-6):
    """S_in máximo que cumple O2 >= frac*O2* en la fase alimentada (X(t_f) crece con S_in)."""
    lim = frac * O2S
    while b - a > tol:
        c = 0.5 * (a + b)
        if o2_min_alimentacion(th, c)[0] >= lim:
            a = c
        else:
            b = c
    return a, o2_min_alimentacion(th, a)[1]


def verificar(th):
    pv = pd.read_excel(XL_FB, "prediccion_vs_medido")
    _, Y = simular(th, S_EXP, pv["t_h"].values.astype(float))
    dif = max(np.max(np.abs(Y[k] - pv[c + "_modelo"].values)) for k, c in enumerate(["X", "S", "P", "O2"]))
    print(f"Comprobación contra resultados_fedbatch.xlsx: diferencia máxima = {dif:.1e} g/L")
    return pv


def figura_optimo(th, s_opt):
    """Sección de optimización: solo modelo, sin mediciones."""
    tt = np.linspace(0.0, 12.0, 1201)
    _, Yo = simular(th, s_opt, tt)
    _, Ye = simular(th, S_EXP, tt)
    fig, axs = plt.subplots(3, 1, figsize=(3.3, 5.2), sharex=True)
    for ax, k, f, etiq in [(axs[0], 0, 1.0, "Biomass (g/L)"), (axs[1], 1, 1.0, "Glucose (g/L)"),
                           (axs[2], 3, 1e3, "Dissolved O$_2$ ($10^{-3}$ g/L)")]:
        ax.axvspan(T_ON, T_OFF, color="0.92", lw=0, label="Feeding")
        ax.plot(tt, f * Yo[k], "-", color="C1", lw=1.4, label=f"$S_\\mathrm{{in}}^*$ = {s_opt:.2f} g/L (optimum)")
        ax.plot(tt, f * Ye[k], "--", color="C0", lw=1.2, label=f"$S_\\mathrm{{in}}$ = {S_EXP:.2f} g/L")
        ax.set_ylabel(etiq)
    axs[2].axhline(1e3 * 0.2 * O2S, color="r", ls=":", lw=0.9, label="$O_{2,\\min}$ (20 % of $O_2^*$)")
    axs[2].set_xlabel("Time (h)")
    axs[2].set_xlim(0, 12)
    axs[0].legend(loc="upper left", frameon=False)
    axs[2].legend(loc="lower left", frameon=False, handles=[axs[2].get_lines()[-1]])
    for ax, letra in zip(axs, "abc"):
        ax.text(0.98, 0.06, f"({letra})", transform=ax.transAxes, ha="right", fontsize=8.5, fontweight="bold")
    fig.tight_layout(h_pad=0.5)
    fig.savefig(os.path.join(AQUI, "fedbatch_optimo.png"), dpi=300)
    plt.close(fig)


def figura_validacion(th, pv):
    """Sección de validación: solo S_in = 15.43 g/L, que es la condición del experimento."""
    tt = np.linspace(0.0, 12.0, 1201)
    _, Ye = simular(th, S_EXP, tt)
    t = pv["t_h"].values
    cal = t <= 5
    val = t >= 6
    etiq = {"X": "Biomass (g/L)", "S": "Glucose (g/L)", "P": "OFP (g/L)", "O2": "Dissolved O$_2$ ($10^{-3}$ g/L)"}
    fig, axs = plt.subplots(2, 2, figsize=(3.3, 3.9))
    axs = axs.ravel()
    for i, v in enumerate(["X", "S", "P", "O2"]):
        ax = axs[i]
        f = 1e3 if v == "O2" else 1.0
        ax.axvspan(0, 5, color="0.92", lw=0, label="Calibration data (0–5 h)")
        ax.axvline(T_ON, color="0.4", ls=":", lw=0.8)
        ax.plot(tt, f * Ye[i], "-", color="C0", lw=1.3, label=f"Model ($S_\\mathrm{{in}}$ = {S_EXP:.2f} g/L)")
        med = pv[v + "_medido"].values
        ax.plot(t[cal], f * med[cal], "o", color="0.55", ms=3.2, label="Measured (calibration)", zorder=5)
        ax.plot(t[val], f * med[val], "o", color="k", ms=3.2, label="Measured (validation)", zorder=5)
        if v == "O2":
            ax.axhline(f * 0.2 * O2S, color="r", ls=":", lw=0.9, label="$O_{2,\\min}$ (20 % of $O_2^*$)")
        ax.set_title(etiq[v])
        ax.set_xlabel("Time (h)")
        ax.set_xlim(0, 12)
    h1, l1 = axs[3].get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.885), w_pad=0.6, h_pad=0.6)
    fig.savefig(os.path.join(AQUI, "fedbatch_validacion.png"), dpi=300)
    plt.close(fig)


def figura_convergencia():
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for k in range(1, 11):
        c = pd.read_excel(XL_FB, f"convergencia_s{k}")
        ax.plot(np.arange(1, len(c) + 1), c["mejor_X"], lw=0.8, alpha=0.85)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best feasible $X(t_f)$ (g/L)")
    fig.tight_layout()
    fig.savefig(os.path.join(AQUI, "fedbatch_convergencia_AG.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    sel = pd.read_excel(XL_CAL, sheet_name="F2_seleccionada")
    th = sel["valor"].iloc[:8].astype(float).values
    th_cheb = sel["valor_chebyshev"].iloc[:8].astype(float).values
    pv = verificar(th)
    s_opt = float(pd.read_excel(XL_FB, "biseccion")["Sin_opt"].iloc[0])

    figura_optimo(th, s_opt)
    figura_validacion(th, pv)
    figura_convergencia()

    filas = []
    for pct in [5, 10, 15, 20, 25, 30]:
        s, x = optimo_por_biseccion(th, pct / 100)
        filas.append(dict(parametros="F2 (seleccionada)", limite_O2_pct=pct, Sin_opt=s, X_tf=x))
    s, x = optimo_por_biseccion(th_cheb, 0.20)
    filas.append(dict(parametros="F2 (Chebyshev)", limite_O2_pct=20, Sin_opt=s, X_tf=x))
    tabla = pd.DataFrame(filas)
    tabla.to_excel(os.path.join(AQUI, "sensibilidad_fedbatch.xlsx"), index=False)
    print(tabla.round(3).to_string(index=False))
    print("Listo: fedbatch_optimo.png, fedbatch_validacion.png, fedbatch_convergencia_AG.png, sensibilidad_fedbatch.xlsx")
