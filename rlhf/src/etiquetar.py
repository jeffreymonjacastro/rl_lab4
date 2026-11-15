"""Parte C - Etiquetado humano de preferencias: elige la evasion mas fluida y segura de cada par.

    python3 rlhf/src/etiquetar.py      # con la figura enfocada: 1 o 2 elige, S salta, Q termina
Cada figura muestra, por opcion, la ruta (x, y), w(t) y la distancia al obstaculo mas cercano.
Criterio usado: llega a la meta > no choca > mayor margen al obstaculo > w mas suave.
Se puede interrumpir y continuar: los pares ya etiquetados no se repiten.
"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
import csv
import numpy as np

DATOS = RAIZ/'rlhf'/'results'/'data'
AZUL, NARANJA, CRITICO = '#2a78d6', '#eb6834', '#d03b3b'


def mostrar(plt, par, a, b):
    fig, ejes = plt.subplots(3, 2, figsize=(12, 8.5), gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    for j, d in enumerate((a, b)):
        fin = 'CHOQUE' if bool(d['colision']) else 'META' if bool(d['exito']) else 'sin terminar'
        t = np.arange(len(d['acciones']))*0.2
        ax = ejes[0, j]
        ax.plot(d['xy'][:, 0], d['xy'][:, 1], '-o', color=AZUL, markersize=3)
        ax.scatter([0], [0], color='black', zorder=3, label='inicio')
        ax.add_patch(plt.Circle(d['meta'], 0.2, color='#0ca30c', alpha=0.25, label='meta'))
        if bool(d['colision']):
            ax.scatter(*d['xy'][-1], marker='X', s=140, color=CRITICO, zorder=4, label='choque')
        ax.set_xlim(-0.4, 2.0); ax.set_ylim(-1.0, 1.0); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_title(f'Opcion {j+1}: {fin} en {t[-1]+0.2:.1f} s', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ejes[1, j].plot(t, d['acciones'][:, 1], color=NARANJA)
        ejes[1, j].set_ylim(-2.9, 2.9); ejes[1, j].set_ylabel('omega (rad/s)'); ejes[1, j].grid(alpha=.3)
        ejes[2, j].plot(t, d['dmin'], color=AZUL)
        ejes[2, j].axhline(0.15, color=CRITICO, linestyle='--', linewidth=1)
        ejes[2, j].set_ylim(0, 1.0); ejes[2, j].set_ylabel('obstaculo mas\ncercano (m)')
        ejes[2, j].set_xlabel('tiempo (s)'); ejes[2, j].grid(alpha=.3)
    fig.suptitle(f'Par {par+1}: ¿cual evade de forma mas fluida y segura?   Teclas: 1 / 2 = elegir,  '
                 'S = saltar,  Q = terminar')
    fig.tight_layout()
    eleccion = {}

    def tecla(evento):
        if evento.key in ('1', '2', 's', 'q'):
            eleccion['k'] = evento.key
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', tecla)
    plt.show()
    return eleccion.get('k', 'q')


def main():
    import matplotlib.pyplot as plt
    archivos = sorted((DATOS/'trayectorias').glob('trayectoria_*.npz'))
    if len(archivos) < 2:
        raise SystemExit('Faltan trayectorias: ejecuta preferencias.py recoger.')
    salida = DATOS/'preferencias.csv'
    hechos = set()
    if salida.exists():
        with open(salida, newline='') as f:
            hechos = {int(fila['par']) for fila in csv.DictReader(f)}
    with open(salida, 'a', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(['par', 'ganadora', 'perdedora', 'origen'])
        for par in range(len(archivos)//2):
            if par in hechos:
                continue
            pa, pb = archivos[2*par], archivos[2*par+1]
            a, b = dict(np.load(pa)), dict(np.load(pb))
            k = mostrar(plt, par, a, b)
            if k == 'q':
                break
            if k == 's':
                continue
            gana, pierde = (pa, pb) if k == '1' else (pb, pa)
            w.writerow([par, gana.name, pierde.name, 'humano'])
            f.flush()
    with open(salida, newline='') as f:
        print(f'{len(list(csv.DictReader(f)))} pares etiquetados en {salida}')


if __name__ == '__main__':
    main()
