"""
NSGA3_calibracion_por_fermentacion.py

Calibración de los 8 parámetros del modelo cinético con NSGA-III (pymoo),
una fermentación a la vez: F1 y F2 tienen cada una su propio conjunto de parámetros.

Modelo (ecuaciones de la tesis, sin cambios):
    mu   = mumax * S/(Ks+S) * O2/(KO2+O2)
    dX/dt  = mu*X
    dS/dt  = -mu*X/Yxs
    dP/dt  = (mu/Yxp + beta)*X
    dO2/dt = KLa*(O2* - O2) - (mu/YxO2 + qO2m)*X

Objetivos (4, se minimizan): f_k = suma de residuos al cuadrado de la variable k
(biomasa, glucosa, OFP, O2) en la fermentación que se calibra. La glucosa medida
como 0 está por debajo del límite de detección: su residuo es max(simulado - S_LOD, 0).

Salidas (en la carpeta del script):
    resultados_NSGA3_por_fermentacion.xlsx
    ajuste_F1.png, ajuste_F2.png, hipervolumen_F1.png, hipervolumen_F2.png

Uso:
    python NSGA3_calibracion_por_fermentacion.py            # corrida completa
    python NSGA3_calibracion_por_fermentacion.py --prueba   # prueba rápida (2 semillas, 10 generaciones)

Probado con Python 3.14, NumPy 2.4, SciPy 1.17, pymoo 0.6.1.6, pandas y openpyxl.
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
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
PRUEBA = "--prueba" in sys.argv
FERMENTACIONES = ["F1", "F2"]
SEMILLAS = [1, 2] if PRUEBA else list(range(1, 11))
POP_SIZE = 100
N_GEN = 10 if PRUEBA else 200
N_PARTITIONS = 6                      # Das-Dennis, 4 objetivos -> 84 direcciones
RTOL, ATOL = 1e-8, 1e-11              # LSODA
N_JOBS = max(1, (os.cpu_count() or 2) - 1)   # procesos en paralelo (una semilla por proceso)
SALIDA = os.path.dirname(os.path.abspath(__file__))

# --- Constantes fijas ---------------------------------------------------------
T_C = 37.0                            # temperatura de operación (°C)
ALTITUD_M = 2286.0                    # Atizapán de Zaragoza (Tec, Campus Estado de México)
P_ATM = (1 - 2.25577e-5 * ALTITUD_M) ** 5.25588   # atmósfera estándar (atm)
Y_O2_AIRE = 0.21
# Ley de Henry; correlación de Perry (8a ed., Tabla 2-123): ln x = A + B/T + C ln T
A_H, B_H, C_H = -171.2542, 8391.24, 23.24323
_T_K = T_C + 273.15
H_ATM = 1.0 / np.exp(A_H + B_H / _T_K + C_H * np.log(_T_K))
O2_SAT = Y_O2_AIRE * P_ATM / H_ATM * (1000.0 / 18.015) * 32.0   # g/L

QO2M = 0.0144                         # g O2 / (g X h), Mainzer y Hempfling (1976)
S_LOD = 0.1                           # g/L, límite de detección del glucómetro

# Los datos de O2 de abajo están en g/L, convertidos de % de saturación con este factor.
# Se vuelven a convertir con O2_SAT para que el modelo y los datos usen el mismo valor.
O2_FACTOR_DATOS = 0.005697485

# --- Biomasa inicial de cada simulación ----------------------------------------
# 'medido': valor medido a 0 h.   'minimo': valor medido más bajo de esa fermentación.
X0_REGLA = {"F1": "minimo", "F2": "minimo"}

# --- Regla para elegir la solución representativa del frente --------------------
# 'rango_medido' : mínima suma de f_k / (rango medido de la variable k)^2
#                  (equivale a minimizar la suma de los NRMSE^2; no depende de los extremos del frente)
# 'minmax_frente': mínima suma con pesos iguales de los objetivos normalizados min-max sobre el frente
#                  (regla original de la sección 2.6)
REGLA_SELECCION = "rango_medido"

# --- Límites de los parámetros (los de la tesis) -------------------------------
NOMBRES = ["mumax", "Ks", "KO2", "Yxs", "Yxp", "beta", "KLa", "YxO2"]
UNIDADES = ["h-1", "g/L", "g/L", "gX/gS", "gX/gP", "gP/(gX h)", "h-1", "gX/gO2"]
XL = np.array([0.22, 0.50, 5e-5, 0.29, 15.23, 4e-4, 100.0, 0.96])
XU = np.array([0.80, 1.00, 1e-4, 0.70, 50.00, 1e-3, 180.0, 1.30])

VARS = ["X", "S", "P", "O2"]
ETIQ = {"X": "Biomasa (g/L)", "S": "Glucosa (g/L)", "P": "OFP (g/L)", "O2": "O$_2$ disuelto (g/L)"}

# --- Datos medidos --------------------------------------------------------------
# F1: corrida batch del 21/02/2024 (Tabla 6.1 de la tesis).
# F2: fase batch (0-5 h) de la corrida fed-batch del 28/08/2024; glucosa en lectura directa.
_DATOS_ORIG = {
    "F1": dict(t=np.arange(8.0),
               X=[0.35, 0.283, 0.367, 0.733, 1.4, 2.1, 2.067, 2.233],
               S=[3.88, 3.68, 3.53, 3.27, 2.38, 0.75, 0.0, 0.0],
               P=[0.008, 0.013, 0.0195, 0.033, 0.04095, 0.0489, 0.0535, 0.0605],
               O2=[0.005680, 0.005452, 0.005116, 0.004450, 0.003783, 0.003686, 0.005128, 0.005481]),
    "F2": dict(t=np.arange(6.0),
               X=[0.583333, 0.416667, 0.466667, 1.55, 2.6, 3.55],
               S=[4.71, 4.48, 4.08, 3.32, 0.88, 0.0],
               P=[0.004709, 0.005102, 0.005494, 0.069070, 0.086338, 0.087908],
               O2=[0.005413, 0.005242, 0.004296, 0.003134, 0.001282, 0.005339]),
}
DATOS = {}
for _n, _d in _DATOS_ORIG.items():
    DATOS[_n] = {k: np.asarray(v, float) for k, v in _d.items()}
    DATOS[_n]["O2"] = DATOS[_n]["O2"] / O2_FACTOR_DATOS * O2_SAT


def x0_de(n):
    d = DATOS[n]
    return float(d["X"][0] if X0_REGLA[n] == "medido" else d["X"].min())


# =============================================================================
# MODELO
# =============================================================================
def rhs(t, y, th):
    mumax, Ks, KO2, Yxs, Yxp, beta, KLa, YxO2 = th
    X, S, P, O = y
    Sp, Op = max(S, 0.0), max(O, 0.0)
    mu = mumax * Sp / (Ks + Sp) * Op / (KO2 + Op)
    return [mu * X,
            -mu * X / Yxs,
            (mu / Yxp + beta) * X,
            KLa * (O2_SAT - O) - (mu / YxO2 + QO2M) * X]


def simular(th, n, t_eval):
    d = DATOS[n]
    y0 = [x0_de(n), d["S"][0], d["P"][0], d["O2"][0]]
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), y0, method="LSODA",
                    t_eval=t_eval, args=(th,), rtol=RTOL, atol=ATOL)
    if not sol.success or sol.y.shape[1] != len(t_eval):
        return None
    return sol.y


def residuos(th, n):
    d = DATOS[n]
    y = simular(th, n, d["t"])
    if y is None:
        return None
    res = {}
    for i, v in enumerate(VARS):
        r = y[i] - d[v]
        if v == "S":
            cens = d["S"] <= 0.0
            r[cens] = np.maximum(y[i][cens] - S_LOD, 0.0)
        res[v] = r
    return res


class Calibracion(ElementwiseProblem):
    def __init__(self, n):
        super().__init__(n_var=8, n_obj=4, xl=XL, xu=XU)
        self.n = n

    def _evaluate(self, x, out, *args, **kwargs):
        res = residuos(x, self.n)
        out["F"] = [1e6] * 4 if res is None else [float(np.sum(res[v] ** 2)) for v in VARS]


class GuardarFrente(Callback):
    """Guarda el conjunto no dominado de cada generación para el hipervolumen."""
    def __init__(self):
        super().__init__()
        self.frentes = []

    def notify(self, algorithm):
        self.frentes.append(algorithm.opt.get("F").copy())


def correr_semilla(args):
    n, seed = args
    t0 = time.time()
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=N_PARTITIONS)
    alg = NSGA3(ref_dirs=ref_dirs, pop_size=POP_SIZE,
                crossover=SBX(eta=30, prob=1.0),   # valores por defecto de NSGA3 en pymoo
                mutation=PM(eta=20),               # prob. por variable por defecto = 1/n_var = 1/8
                eliminate_duplicates=True)
    cb = GuardarFrente()
    res = minimize(Calibracion(n), alg, ("n_gen", N_GEN), seed=seed, callback=cb, verbose=False)
    return n, seed, res.pop.get("X"), res.pop.get("F"), cb.frentes, time.time() - t0


# =============================================================================
# POSPROCESO
# =============================================================================
def no_dominadas(F):
    return NonDominatedSorting().do(F, only_non_dominated_front=True)


def normalizar(F, n, regla):
    if regla == "rango_medido":
        rango = np.array([DATOS[n][v].max() - DATOS[n][v].min() for v in VARS])
        return F / rango ** 2
    fmin, fmax = F.min(axis=0), F.max(axis=0)
    return (F - fmin) / np.where(fmax > fmin, fmax - fmin, 1.0)


def seleccionar(F, n, regla=None):
    """Devuelve (índice con la suma mínima de objetivos normalizados,
    índice con el criterio min-max de Chebyshev como comprobación)."""
    Fn = normalizar(F, n, regla or REGLA_SELECCION)
    return int(np.argmin(Fn.sum(axis=1))), int(np.argmin(Fn.max(axis=1)))


def metricas(th, n):
    d = DATOS[n]
    y = simular(th, n, d["t"])
    filas = []
    for i, v in enumerate(VARS):
        med, sim = d[v], y[i]          # glucosa bajo el límite cuenta como 0
        e = sim - med
        rmse = np.sqrt(np.mean(e ** 2))
        rango = med.max() - med.min()
        filas.append(dict(variable=v,
                          R2=1 - np.sum(e ** 2) / np.sum((med - med.mean()) ** 2),
                          RMSE=rmse, NRMSE=rmse / rango, MBE=e.mean(), error_max=np.abs(e).max()))
    return pd.DataFrame(filas)


def graficar_ajuste(th, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = DATOS[n]
    t_fin = 6.0 if n == "F2" else float(d["t"][-1])
    tt = np.linspace(0, t_fin, 361)
    y = simular(th, n, tt)
    m = metricas(th, n)
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4))
    for i, v in enumerate(VARS):
        ax = axs[i]
        if n == "F2":
            ax.axvspan(5.5, 6.0, color="0.9", zorder=0, label="Inicio de la alimentación (5.5 h)")
        ax.plot(tt, y[i], "-", lw=2.2, label="Modelo (NSGA-III)")
        ax.plot(d["t"], d[v], "o", c="k", ms=6.5, label="Medido", zorder=5)
        ax.set_title(f"{n} — {ETIQ[v]}\nR² = {m.loc[i, 'R2']:.3f}", fontsize=10)
        ax.set_xlabel("Tiempo (h)")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, f"ajuste_{n}.png"), dpi=200)
    plt.close(fig)


def graficar_hv(hv_df, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c in hv_df.columns:
        ax.plot(hv_df.index, hv_df[c], lw=1, alpha=0.8)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Hipervolumen (objetivos normalizados)")
    ax.set_title(f"{n}: convergencia de las {hv_df.shape[1]} semillas")
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, f"hipervolumen_{n}.png"), dpi=200)
    plt.close(fig)


def procesar(n, corridas):
    # Frente final: soluciones no dominadas de las poblaciones finales de todas las semillas
    X_all = np.vstack([c[2] for c in corridas])
    F_all = np.vstack([c[3] for c in corridas])
    nd = no_dominadas(F_all)
    Xf, Ff = X_all[nd], F_all[nd]
    i_sel, i_cheb = seleccionar(Ff, n)
    th = Xf[i_sel]

    # Hipervolumen: objetivos normalizados con el rango (ideal y nadir) del frente final;
    # punto de referencia 1.1 en cada eje.
    ideal, nadir = Ff.min(axis=0), Ff.max(axis=0)
    escala = np.where(nadir > ideal, nadir - ideal, 1.0)
    hv = HV(ref_point=np.full(4, 1.1))
    hv_df = pd.DataFrame({f"semilla_{c[1]}": [hv((Fg - ideal) / escala) for Fg in c[4]]
                          for c in corridas})
    hv_df.index = np.arange(1, len(hv_df) + 1)
    hv_df.index.name = "generacion"

    # Variabilidad entre semillas: misma regla aplicada al frente de cada semilla
    filas = []
    for c in corridas:
        nd_s = no_dominadas(c[3])
        j, _ = seleccionar(c[3][nd_s], n)
        filas.append(dict(semilla=c[1], **dict(zip(NOMBRES, c[2][nd_s][j])),
                          tiempo_min=c[5] / 60))
    sem = pd.DataFrame(filas)
    cv = (sem[NOMBRES].std(ddof=1) / sem[NOMBRES].mean() * 100).to_dict()
    sem = pd.concat([sem, pd.DataFrame([dict(semilla="CV (%)", **cv)])], ignore_index=True)

    frente = pd.DataFrame(Xf, columns=NOMBRES)
    for k, v in enumerate(VARS):
        frente[f"f_{v}"] = Ff[:, k]

    ancho = 0.01 * (XU - XL)
    sel = pd.DataFrame(dict(parametro=NOMBRES, unidades=UNIDADES, valor=th,
                            limite_inferior=XL, limite_superior=XU,
                            en_el_limite=[("inferior" if th[k] - XL[k] < ancho[k] else
                                           "superior" if XU[k] - th[k] < ancho[k] else "no")
                                          for k in range(8)],
                            valor_chebyshev=Xf[i_cheb]))
    extra = pd.DataFrame(dict(parametro=["X0 usado (g/L)", "misma solucion con Chebyshev",
                                         "soluciones en el frente final"],
                              valor=[x0_de(n), "si" if i_sel == i_cheb else "no", len(Xf)]))
    # Comparación de las dos reglas de selección sobre el mismo frente
    comp = []
    for regla in ["rango_medido", "minmax_frente"]:
        k, _ = seleccionar(Ff, n, regla)
        mt = metricas(Xf[k], n)
        comp.append(dict(regla=regla + (" (usada)" if regla == REGLA_SELECCION else ""),
                         **{f"R2_{v}": r for v, r in zip(mt.variable, mt.R2)},
                         **dict(zip(NOMBRES, Xf[k]))))
    graficar_ajuste(th, n)
    graficar_hv(hv_df, n)
    return dict(frente=frente, seleccionada=pd.concat([sel, extra], ignore_index=True),
                ajuste=metricas(th, n), por_semilla=sem, hipervolumen=hv_df,
                reglas=pd.DataFrame(comp))


def tabla_configuracion():
    filas = [
        ("O2* (g/L)", O2_SAT), ("Presión (atm), atmósfera estándar", P_ATM), ("Altitud (m)", ALTITUD_M),
        ("Temperatura (°C)", T_C), ("H de Henry (atm)", H_ATM), ("qO2m (gO2/gX/h)", QO2M),
        ("Límite de detección de glucosa (g/L)", S_LOD), ("Factor con que se convirtieron los datos de O2", O2_FACTOR_DATOS),
        ("X0 F1", f"{x0_de('F1')} ({X0_REGLA['F1']})"), ("X0 F2", f"{x0_de('F2')} ({X0_REGLA['F2']})"),
        ("Población", POP_SIZE), ("Generaciones", N_GEN), ("Particiones Das-Dennis", N_PARTITIONS),
        ("Semillas", str(SEMILLAS)), ("Cruza", "SBX, prob 1.0, eta 30"),
        ("Mutación", "polinomial, eta 20, prob. por individuo 0.9, prob. por variable 1/8"),
        ("Integrador", f"LSODA, rtol {RTOL}, atol {ATOL}"),
        ("Regla de selección", REGLA_SELECCION),
    ]
    return pd.DataFrame(filas, columns=["concepto", "valor"])


# =============================================================================
if __name__ == "__main__":
    print(f"O2* = {O2_SAT:.6f} g/L (P = {P_ATM:.4f} atm, {T_C} °C) | X0: F1 = {x0_de('F1'):.4f}, F2 = {x0_de('F2'):.4f}")
    print(f"{len(FERMENTACIONES)} fermentaciones x {len(SEMILLAS)} semillas, {N_GEN} generaciones, {N_JOBS} procesos")
    tareas = [(n, s) for n in FERMENTACIONES for s in SEMILLAS]
    corridas = {n: [] for n in FERMENTACIONES}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        for n, seed, X, F, frentes, dt in ex.map(correr_semilla, tareas):
            corridas[n].append((n, seed, X, F, frentes, dt))
            print(f"  {n} semilla {seed:2d} terminada en {dt / 60:.1f} min")
    print(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")

    ruta = os.path.join(SALIDA, "resultados_NSGA3_por_fermentacion.xlsx")
    with pd.ExcelWriter(ruta) as xw:
        tabla_configuracion().to_excel(xw, sheet_name="configuracion", index=False)
        for n in FERMENTACIONES:
            r = procesar(n, sorted(corridas[n], key=lambda c: c[1]))
            r["seleccionada"].to_excel(xw, sheet_name=f"{n}_seleccionada", index=False)
            r["ajuste"].to_excel(xw, sheet_name=f"{n}_ajuste", index=False)
            r["frente"].to_excel(xw, sheet_name=f"{n}_frente_final", index=False)
            r["por_semilla"].to_excel(xw, sheet_name=f"{n}_por_semilla", index=False)
            r["hipervolumen"].to_excel(xw, sheet_name=f"{n}_hipervolumen")
            r["reglas"].to_excel(xw, sheet_name=f"{n}_reglas_seleccion", index=False)
            print(f"\n=== {n} ===")
            print(r["seleccionada"].to_string(index=False))
            print(r["ajuste"].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
            print(r["reglas"][["regla"] + [f"R2_{v}" for v in VARS]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nResultados en {ruta}")
