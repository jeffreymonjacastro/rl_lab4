"""Estilo comun de las graficas y lectura de CSV."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AZUL, NARANJA, CRITICO, BUENO, TINTA, TENUE = '#2a78d6', '#eb6834', '#d03b3b', '#0ca30c', '#2b2b28', '#8a8980'

plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': '#c9c8c0', 'axes.labelcolor': TINTA, 'text.color': TINTA,
    'xtick.color': TENUE, 'ytick.color': TENUE, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5df',
    'grid.linewidth': 0.8, 'lines.linewidth': 2.0, 'font.size': 10,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'legend.frameon': False,
})


def filas(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def guardar(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print('Grafica:', path)
