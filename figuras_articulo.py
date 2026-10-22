"""
Figuras del artículo (en inglés) a partir de los resultados ya calculados.
No vuelve a correr NSGA-III: lee resultados_NSGA3_por_fermentacion.xlsx y usa el modelo
de NSGA3_calibracion_por_fermentacion.py. Ambos archivos deben estar en la misma carpeta.

Salidas (300 dpi), con los nombres que usa el artículo:
    ajuste_F1.png, ajuste_F2.png               ajuste de cada fermentación (1 x 4 paneles)
    hipervolumen_F1.png, hipervolumen_F2.png   convergencia del hipervolumen de cada fermentación

Uso:
    python figuras_articulo.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
_argv = sys.argv
sys.argv = [sys.argv[0]]                       # que el script de calibración no lea estos argumentos
import NSGA3_calibracion_por_fermentacion as cal
sys.argv = _argv

EXCEL = os.path.join(AQUI, "resultados_NSGA3_por_fermentacion.xlsx")
FERM = ["F1", "F2"]
ETIQ = {"X": "Biomass (g/L)", "S": "Glucose (g/L)", "P": "OFP (g/L)", "O2": "Dissolved O$_2$ ($10^{-3}$ g/L)"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 7.5,
                     "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.6})


def parametros(n):
    sel = pd.read_excel(EXCEL, sheet_name=f"{n}_seleccionada")
    return sel["valor"].iloc[:8].astype(float).values


def figura_ajuste(n, letra):
    """Una figura por fermentación (1 x 4 paneles), con el nombre que usa el artículo."""
    th = parametros(n)
    d = cal.DATOS[n]
    tt = np.linspace(0.0, float(d["t"][-1]), 301)
    y = cal.simular(th, n, tt)
    m = cal.metricas(th, n)
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 2.15))
    for i, v in enumerate(cal.VARS):
        ax = axs[i]
        f = 1e3 if v == "O2" else 1.0           # oxígeno en 10^-3 g/L, como en las tablas
        ax.plot(tt, f * y[i], "-", lw=1.4, color="C0", label="Model")
        ax.plot(d["t"], f * d[v], "o", color="k", ms=3.5, label="Measured", zorder=5)
        ax.set_title(f"{ETIQ[v]}\n$R^2$ = {m.loc[i, 'R2']:.3f}")
        ax.set_xlabel("Time (h)")
    axs[0].text(-0.42, 1.18, f"({letra}) {n}", transform=axs[0].transAxes,
                fontsize=8.5, fontweight="bold", va="bottom")
    axs[0].legend(loc="upper left", frameon=False)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(AQUI, f"ajuste_{n}.png"), dpi=300)
    plt.close(fig)


def figura_hipervolumen(n, letra):
    """Una figura por fermentación, con el nombre que usa el artículo."""
    hv = pd.read_excel(EXCEL, sheet_name=f"{n}_hipervolumen", index_col=0)
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for c in hv.columns:
        ax.plot(hv.index, hv[c], lw=0.7, alpha=0.85)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized hypervolume")
    ax.text(0.03, 0.07, f"({letra}) {n}", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(AQUI, f"hipervolumen_{n}.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    for n, letra in zip(FERM, "ab"):
        figura_ajuste(n, letra)
        figura_hipervolumen(n, letra)
    print("Listo: ajuste_F1.png, ajuste_F2.png, hipervolumen_F1.png, hipervolumen_F2.png")
