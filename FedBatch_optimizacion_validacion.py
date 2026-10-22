"""
FedBatch_optimizacion_validacion.py

Optimización del fed-batch con un algoritmo genético (AG) y validación predictiva,
usando los parámetros de F2 calibrados con NSGA3_calibracion_por_fermentacion.py.

Requiere en la misma carpeta:
    NSGA3_calibracion_por_fermentacion.py   (de ahí se toman O2*, qO2m, datos y X0, para que
                                             la calibración y el fed-batch usen exactamente lo mismo)
    resultados_NSGA3_por_fermentacion.xlsx  (hoja "F2_seleccionada", generada por la calibración)

Modelo fed-batch (mismas ecuaciones que el batch, con dilución D = F/V):
    dX/dt  = mu*X - D*X
    dS/dt  = D*(Sin - S) - mu*X/Yxs
    dP/dt  = (mu/Yxp + beta)*X - D*P
    dO2/dt = KLa*(O2* - O2) - (mu/YxO2 + qO2m)*X - D*O2
    dV/dt  = F

Problema: maximizar X(t_f) con la concentración de glucosa en la alimentación (Sin) como
variable de decisión, sujeto a (solo en la fase con alimentación, t >= 5.5 h):
    g1 = max V - 2.05 <= 0
    g2 = 0.2*O2* - min O2 <= 0

Salidas: resultados_fedbatch.xlsx, fedbatch_validacion.png, fedbatch_convergencia_AG.png

Uso:
    python FedBatch_optimizacion_validacion.py            # corrida completa
    python FedBatch_optimizacion_validacion.py --prueba   # prueba rápida
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

import NSGA3_calibracion_por_fermentacion as cal

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
PRUEBA = "--prueba" in sys.argv
SALIDA = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_CALIBRACION = os.path.join(SALIDA, "resultados_NSGA3_por_fermentacion.xlsx")
FERMENTACION = "F2"                   # el fed-batch es la misma corrida que F2

SEMILLAS = [1, 2] if PRUEBA else list(range(1, 11))
POP_SIZE = 10 if PRUEBA else 50
N_GEN = 5 if PRUEBA else 100
N_JOBS = max(1, (os.cpu_count() or 2) - 1)
RTOL, ATOL = 1e-9, 1e-12

# --- Operación ----------------------------------------------------------------
V0 = 1.5                              # L
F_ALIM = 4.5 * 60 / 1000              # 4.5 mL/min = 0.27 L/h
T_ALIM = 5.5                          # h, inicio de la alimentación
V_MAX = 2.0                           # L, la alimentación se detiene al llegar aquí
T_FIN_ALIM = T_ALIM + (V_MAX - V0) / F_ALIM   # ~7.35 h
T_FINAL = 20.0                        # h, horizonte de la optimización
DT = 0.01                             # h, malla para evaluar las restricciones
V_LIMITE = 2.05                       # L
O2_MIN = 0.2 * cal.O2_SAT             # 20 % de saturación
SIN_LIM = (0.0, 100.0)                # g/L
SIN_EXPERIMENTAL = 15.43              # g/L, el que se usó en el experimento

# --- Datos medidos de la fase con alimentación (6-11 h) -----------------------
# Biomasa, OFP y O2: Tabla 6.18 de la tesis (4 decimales; el redondeo del O2 es <= 5e-5 g/L,
# menos del 1 % de saturación). Glucosa: lectura directa del glucómetro (0 = por debajo de
# 0.1 g/L). El O2 está en g/L convertido con 0.005697485 y se vuelve a convertir con el O2*
# de la calibración.
VALIDACION = dict(
    t=np.arange(6.0, 12.0),
    X=np.array([3.6333, 4.1000, 4.6000, 4.7167, 4.5500, 4.3167]),
    S=np.array([0.30, 0.30, 0.0, 0.0, 0.0, 0.0]),
    P=np.array([0.1052, 0.1056, 0.1268, 0.1381, 0.1491, 0.1601]),
    O2=np.array([0.0036, 0.0019, 0.0026, 0.0055, 0.0056, 0.0057]) / cal.O2_FACTOR_DATOS * cal.O2_SAT,
)


# =============================================================================
# PARÁMETROS Y CONDICIONES INICIALES
# =============================================================================
def leer_parametros(ferm=FERMENTACION):
    df = pd.read_excel(ARCHIVO_CALIBRACION, sheet_name=f"{ferm}_seleccionada")
    df = df[df["parametro"].isin(cal.NOMBRES)].set_index("parametro")
    return np.array([float(df.loc[n, "valor"]) for n in cal.NOMBRES])


TH = leer_parametros()
_d = cal.DATOS[FERMENTACION]
Y0 = [cal.x0_de(FERMENTACION), _d["S"][0], _d["P"][0], _d["O2"][0], V0]


# =============================================================================
# MODELO
# =============================================================================
def rhs(t, y, Sin, Fc, th):
    mumax, Ks, KO2, Yxs, Yxp, beta, KLa, YxO2 = th
    X, S, P, O, V = y
    Sp, Op = max(S, 0.0), max(O, 0.0)
    mu = mumax * Sp / (Ks + Sp) * Op / (KO2 + Op)
    D = Fc / V
    return [mu * X - D * X,
            D * (Sin - S) - mu * X / Yxs,
            (mu / Yxp + beta) * X - D * P,
            KLa * (cal.O2_SAT - O) - (mu / YxO2 + cal.QO2M) * X - D * O,
            Fc]


def simular(Sin, th=TH, t_final=T_FINAL):
    """Integra por tramos: sin flujo (0-5.5 h), flujo F hasta V = 2.0 L, sin flujo."""
    malla = np.round(np.arange(0.0, t_final + DT / 2, DT), 10)
    tramos = [(0.0, T_ALIM, 0.0), (T_ALIM, T_FIN_ALIM, F_ALIM), (T_FIN_ALIM, t_final, 0.0)]
    y0, T, Y = list(Y0), [], []
    for k, (a, b, Fc) in enumerate(tramos):
        if b <= a:
            continue
        te = np.unique(np.concatenate(([a], malla[(malla >= a) & (malla <= b)], [b])))
        sol = solve_ivp(rhs, (a, b), y0, t_eval=te, args=(Sin, Fc, th),
                        method="LSODA", rtol=RTOL, atol=ATOL)
        if not sol.success:
            raise RuntimeError(f"Integración fallida en el tramo {k}: {sol.message}")
        y0 = sol.y[:, -1]
        ini = 0 if not T else 1
        T.append(sol.t[ini:]); Y.append(sol.y[:, ini:])
    return np.concatenate(T), np.concatenate(Y, axis=1)


def evaluar(Sin):
    t, y = simular(Sin)
    alim = t >= T_ALIM
    return dict(X_final=y[0, -1], X_max=y[0].max(),
                O2_min_alim=y[3][alim].min(), V_max=y[4].max(),
                g_vol=y[4][alim].max() - V_LIMITE, g_O2=O2_MIN - y[3][alim].min())


# =============================================================================
# ALGORITMO GENÉTICO
# =============================================================================
class FedBatch(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=1, n_ieq_constr=2,
                         xl=np.array([SIN_LIM[0]]), xu=np.array([SIN_LIM[1]]))

    def _evaluate(self, x, out, *args, **kwargs):
        r = evaluar(float(x[0]))
        out["F"] = -r["X_final"]
        out["G"] = np.array([r["g_vol"], r["g_O2"]])


class Historia(Callback):
    def __init__(self):
        super().__init__()
        self.data["mejor_X"], self.data["mejor_Sin"], self.data["n_factibles"] = [], [], []

    def notify(self, algorithm):
        o = algorithm.opt[0]
        self.data["mejor_X"].append(-o.F[0] if o.CV[0] <= 0 else np.nan)
        self.data["mejor_Sin"].append(float(o.X[0]))
        self.data["n_factibles"].append(int((algorithm.pop.get("CV")[:, 0] <= 0).sum()))


def correr_ag(seed):
    t0 = time.time()
    cb = Historia()
    alg = GA(pop_size=POP_SIZE, crossover=SBX(prob=0.9, eta=15), mutation=PM(prob=0.9, eta=20),
             eliminate_duplicates=True)   # valores por defecto del GA de pymoo
    res = minimize(FedBatch(), alg, ("n_gen", N_GEN), seed=seed, callback=cb, verbose=False)
    Sin = None if res.X is None else float(np.atleast_1d(res.X)[0])
    return seed, Sin, cb.data, time.time() - t0


def biseccion(tol=1e-6):
    """Mayor Sin que cumple la restricción de O2 (X(tf) crece con Sin)."""
    lo, hi = SIN_LIM[0], 30.0
    if evaluar(lo)["g_O2"] > 0:
        raise RuntimeError("Ni con Sin = 0 se cumple la restricción de O2")
    while evaluar(hi)["g_O2"] <= 0:
        if hi >= SIN_LIM[1]:
            return SIN_LIM[1]        # la restricción no se activa en todo el intervalo
        hi = min(2 * hi, SIN_LIM[1])
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if evaluar(mid)["g_O2"] <= 0 else (lo, mid)
    return lo


def metricas_validacion(t, y):
    filas = []
    for i, v in enumerate(cal.VARS):
        med = VALIDACION[v]
        sim = np.interp(VALIDACION["t"], t, y[i])
        e = sim - med                 # glucosa bajo el límite cuenta como 0
        rmse = np.sqrt(np.mean(e ** 2))
        rango = med.max() - med.min()
        filas.append(dict(variable=v, n=len(med), RMSE=rmse,
                          NRMSE=rmse / rango if rango > 0 else np.nan,
                          MBE=e.mean(), error_max=np.abs(e).max()))
    return pd.DataFrame(filas)


def graficar(t_exp, y_exp, t_opt, y_opt, Sin_opt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = cal.DATOS[FERMENTACION]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (i, v) in zip(axs.flat, enumerate(cal.VARS)):
        ax.axvspan(0, 5.0, color="0.93", zorder=0, label="Calibración (0-5 h)")
        ax.plot(t_exp, y_exp[i], "-", lw=2, label=f"Modelo, Sin = {SIN_EXPERIMENTAL} g/L")
        ax.plot(t_opt, y_opt[i], "--", lw=1.5, label=f"Modelo, Sin* = {Sin_opt:.2f} g/L")
        ax.plot(d["t"], d[v], "o", c="0.4", label="Medido (calibración)")
        ax.plot(VALIDACION["t"], VALIDACION[v], "o", c="k", label="Medido (validación)")
        ax.axvline(T_ALIM, color="gray", lw=0.8, ls=":")
        if v == "O2":
            ax.axhline(O2_MIN, color="red", lw=0.8, ls=":", label="O$_{2,min}$ (20 %)")
        ax.set_xlim(0, 12); ax.set_xlabel("Tiempo (h)"); ax.set_ylabel(cal.ETIQ[v])
    hs, ls = [], []
    for ax in axs.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h); ls.append(l)
    fig.legend(hs, ls, loc="upper center", ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(SALIDA, "fedbatch_validacion.png"), dpi=200)
    plt.close(fig)


def graficar_convergencia(historias):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s, h in historias.items():
        ax.plot(range(1, len(h["mejor_X"]) + 1), h["mejor_X"], lw=1, alpha=0.8)
    ax.set_xlabel("Generación"); ax.set_ylabel("Mejor X(t$_f$) factible (g/L)")
    fig.tight_layout(); fig.savefig(os.path.join(SALIDA, "fedbatch_convergencia_AG.png"), dpi=200)
    plt.close(fig)


# =============================================================================
if __name__ == "__main__":
    print("Parámetros de", FERMENTACION, dict(zip(cal.NOMBRES, np.round(TH, 6))))
    print(f"O2* = {cal.O2_SAT:.6f} g/L | O2_min = {O2_MIN:.6f} g/L | X0 = {Y0[0]:.4f} | "
          f"F = {F_ALIM} L/h | fin de la alimentación = {T_FIN_ALIM:.3f} h")

    # --- AG ------------------------------------------------------------------
    filas, historias = [], {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        for seed, Sin, hist, dt in ex.map(correr_ag, SEMILLAS):
            historias[seed] = hist
            if Sin is None:
                print(f"  semilla {seed}: sin solución factible"); continue
            r = evaluar(Sin)
            filas.append(dict(semilla=seed, Sin_opt=Sin, tiempo_min=dt / 60, **r))
            print(f"  semilla {seed:2d}: Sin* = {Sin:.4f} g/L | X(tf) = {r['X_final']:.4f} g/L | "
                  f"min O2 (alim.) = {r['O2_min_alim']:.6f} | {dt / 60:.1f} min")
    ag = pd.DataFrame(filas)
    print(f"Tiempo del AG: {(time.time() - t0) / 60:.1f} min")

    # --- Verificación independiente ------------------------------------------
    Sin_bis = biseccion()
    r_bis = evaluar(Sin_bis)
    print(f"\nBisección: Sin* = {Sin_bis:.4f} g/L | X(tf) = {r_bis['X_final']:.4f} g/L")
    s_max = max(30.0, 1.5 * Sin_bis)
    barrido = pd.DataFrame([dict(Sin=s, **evaluar(s)) for s in np.arange(0, s_max + 1e-9, 0.5)])
    barrido["factible"] = (barrido["g_O2"] <= 0) & (barrido["g_vol"] <= 0)
    print("¿X(tf) crece con Sin en el barrido?", bool(np.all(np.diff(barrido["X_final"]) >= -1e-9)))

    # --- Validación con el Sin del experimento -------------------------------
    t_e, y_e = simular(SIN_EXPERIMENTAL)
    met = metricas_validacion(t_e, y_e)
    print(f"\nValidación (Sin = {SIN_EXPERIMENTAL} g/L, 6-11 h):")
    print(met.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # --- Guardar ---------------------------------------------------------------
    # Óptimo reportado: la semilla con mayor X(tf) factible; la bisección es la comprobación
    fact = ag[(ag["g_O2"] <= 0) & (ag["g_vol"] <= 0)] if len(ag) else ag
    Sin_ref = float(fact.loc[fact["X_final"].idxmax(), "Sin_opt"]) if len(fact) else Sin_bis
    print(f"Óptimo del AG (mejor semilla): Sin* = {Sin_ref:.4f} g/L | diferencia con la bisección: "
          f"{100 * (Sin_ref - Sin_bis) / Sin_bis:+.3f} %")
    t_o, y_o = simular(Sin_ref)
    horas = np.arange(0.0, 12.0)
    d = cal.DATOS[FERMENTACION]
    comp = pd.DataFrame({"t_h": horas})
    for i, v in enumerate(cal.VARS):
        comp[f"{v}_medido"] = np.concatenate([d[v], VALIDACION[v]])
        comp[f"{v}_modelo"] = np.interp(horas, t_e, y_e[i])
    config = pd.DataFrame([
        ("Parámetros", f"{FERMENTACION}_seleccionada de {os.path.basename(ARCHIVO_CALIBRACION)}"),
        ("O2* (g/L)", cal.O2_SAT), ("qO2m", cal.QO2M), ("O2_min (g/L)", O2_MIN),
        ("X0, S0, P0, O2_0, V0", str(np.round(Y0, 6).tolist())),
        ("F (L/h)", F_ALIM), ("Inicio / fin de la alimentación (h)", f"{T_ALIM} / {T_FIN_ALIM:.4f}"),
        ("t_f (h)", T_FINAL), ("Sin (g/L)", str(SIN_LIM)), ("Malla de restricciones (h)", DT),
        ("AG", f"población {POP_SIZE}, {N_GEN} generaciones, SBX (0.9, eta 15), PM (0.9, eta 20), semillas {SEMILLAS}"),
        ("Integrador", f"LSODA, rtol {RTOL}, atol {ATOL}"),
        ("NRMSE de la validación", "RMSE / rango medido en 6-11 h"),
    ], columns=["concepto", "valor"])
    with pd.ExcelWriter(os.path.join(SALIDA, "resultados_fedbatch.xlsx")) as w:
        config.to_excel(w, sheet_name="configuracion", index=False)
        pd.DataFrame(dict(parametro=cal.NOMBRES, valor=TH)).to_excel(w, sheet_name="parametros", index=False)
        ag.to_excel(w, sheet_name="AG", index=False)
        pd.DataFrame([dict(metodo="biseccion", Sin_opt=Sin_bis, **r_bis)]).to_excel(w, sheet_name="biseccion", index=False)
        barrido.to_excel(w, sheet_name="barrido_Sin", index=False)
        comp.to_excel(w, sheet_name="prediccion_vs_medido", index=False)
        met.to_excel(w, sheet_name="metricas_validacion", index=False)
        pd.DataFrame({"t_h": t_o, **{v: y_o[i] for i, v in enumerate(cal.VARS)}, "V": y_o[4]}
                     ).iloc[::10].to_excel(w, sheet_name="trayectoria_optimo", index=False)
        for s, h in historias.items():
            pd.DataFrame(h).to_excel(w, sheet_name=f"convergencia_s{s}", index=False)
    graficar(t_e, y_e, t_o, y_o, Sin_ref)
    graficar_convergencia(historias)
    print(f"\nResultados en {os.path.join(SALIDA, 'resultados_fedbatch.xlsx')}")
