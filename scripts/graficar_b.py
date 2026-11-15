"""Parte B - Graficas: (3) tasas de intervencion y colision por ronda y (4) mapa 2D de trayectorias.

    python3 scripts/graficar_b.py [--salida carpeta]
"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import argparse
import filecmp
from graficas import plt, filas, guardar, AZUL, NARANJA, CRITICO, BUENO, TENUE

LOGS = RAIZ/'dagger'/'results'/'logs'
BC_LOGS = RAIZ/'bc'/'results'/'logs'
META = (1.5, 0.0)  # escenario cruce (base/entorno.py)
PESOS = RAIZ/'dagger'/'results'/'weights'


def tasas(salida):
    evaluacion = {f['politica']: f for f in filas(LOGS/'dagger_evaluacion.csv')}  # ultima fila por politica
    rondas = {int(f['ronda']): f for f in filas(LOGS/'rondas.csv')}
    if not rondas:
        raise SystemExit('No hay rondas completas de DAgger para graficar.')
    n = max(rondas)
    nombres = ['bc']+[f'dagger_{i}' for i in range(1, n+1)]
    etiquetas = ['BC']+[f'DAgger {i}' for i in range(1, n+1)]
    colision = [float(evaluacion[k]['tasa_colision'])*100 if k in evaluacion else None for k in nombres]
    # La intervencion que necesita una politica se mide en la ronda siguiente, cuando ella pilotea.
    interv = [int(rondas[i+1]['pasos_humanos'])/max(int(rondas[i+1]['pasos']), 1)*100
              if i+1 in rondas else None for i in range(n)]
    fig, ejes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.6))
    for ax, y, color, titulo in [
            (ejes[0], interv, AZUL, 'Tasa de intervencion humana (% de pasos con el humano al mando)'),
            (ejes[1], colision, NARANJA, 'Tasa de colision en evaluacion autonoma (% de episodios)')]:
        x = [i for i, v in enumerate(y) if v is not None]
        v = [y[i] for i in x]
        ax.plot(x, v, '-o', color=color, markersize=7)
        for xi, vi in zip(x, v):
            ax.annotate(f'{vi:.0f}%', (xi, vi), textcoords='offset points', xytext=(0, 8), ha='center')
        ax.set_ylim(0, max(v+[10])*1.3)
        ax.set_title(titulo, loc='left')
    ejes[1].set_xticks(range(n+1))
    ejes[1].set_xticklabels(etiquetas)
    ejes[1].set_xlabel('Politica que pilotea (la ultima no tiene ronda de intervencion posterior)')
    guardar(fig, salida/'dagger_tasas.png')


def trayectorias(salida):
    evaluacion = filas(LOGS/'dagger_evaluacion.csv')
    rondas = sorted({int(f['ronda']) for f in evaluacion} - {0})
    # La ronda del mapa es la que esta copiada en dagger_sim.pth (la entregada); si no, la ultima.
    sim = PESOS/'dagger_sim.pth'
    iguales = [r for r in rondas if sim.exists() and (PESOS/f'dagger_{r}.pth').exists()
               and filecmp.cmp(sim, PESOS/f'dagger_{r}.pth', shallow=False)]
    final = (iguales or rondas or [0])[-1]
    if final == 0:
        return print('Aun no hay rondas de DAgger evaluadas: se omite el mapa de trayectorias.')
    meta = META
    fig, ejes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11, 5))
    puntos = [(0., 0.), meta]
    for ax, nombre, titulo in [(ejes[0], 'bc', 'Behavior Cloning'), (ejes[1], f'dagger_{final}',
                                                                      f'DAgger (ronda {final})')]:
        # mapa_* = evaluacion larga de dagger.py --mapa; si no existe, la de 10 episodios de cada ronda.
        logs = BC_LOGS if nombre == 'bc' else LOGS
        base = 'mapa' if (logs/f'mapa_{nombre}_pasos.csv').exists() else 'eval'
        pasos = filas(logs/f'{base}_{nombre}_pasos.csv')
        fin = {f['episodio']: f for f in filas(logs/f'{base}_{nombre}_episodios.csv')}
        ax.add_patch(plt.Circle(meta, 0.2, color=BUENO, alpha=0.22))
        ax.annotate('meta', meta, ha='center', va='center', fontsize=9)
        ax.scatter([0], [0], color='black', s=30, zorder=3)
        ax.annotate('inicio', (0, 0), textcoords='offset points', xytext=(-6, 8), ha='right', fontsize=9)
        choques = 0
        for ep, f in fin.items():
            xy = [(float(q['x']), float(q['y'])) for q in pasos if q['episodio'] == ep]
            if not xy:
                continue
            puntos.extend(xy)
            choco = f['colision'] == 'True'
            llego = f['exito'] == 'True'
            ax.plot(*zip(*xy), color=AZUL if llego else TENUE, linewidth=1.6, alpha=0.8)
            if choco:
                choques += 1
                ax.scatter(*xy[-1], marker='X', s=90, color=CRITICO, zorder=4)
        exitos = sum(f['exito'] == 'True' for f in fin.values())
        ax.set_title(f'{titulo}: {exitos}/{len(fin)} llegan a la meta, {choques} choques (X)')
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
    ejes[0].set_ylabel('y (m)')
    xs, ys = zip(*puntos)
    ejes[0].set_xlim(min(xs)-0.3, max(xs)+0.3)
    ejes[0].set_ylim(min(ys)-0.3, max(ys)+0.3)
    fig.suptitle('Trayectorias (x, y) de /odom con perturbacion inicial: mismos episodios para ambas politicas',
                 fontsize=10)
    guardar(fig, salida/'trayectorias_bc_vs_dagger.png')


def distribution_shift(salida):
    archivo = BC_LOGS/'bc_distribution_shift.csv'
    if not archivo.exists():
        print('Falta bc_distribution_shift.csv; ejecuta bc.py evaluar para comparar nominal/perturbado.')
        return
    datos = filas(archivo)
    x = list(range(len(datos)))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([i-0.18 for i in x], [100*float(f['tasa_exito']) for f in datos],
           width=0.36, color=BUENO, label='Exito')
    ax.bar([i+0.18 for i in x], [100*float(f['tasa_colision']) for f in datos],
           width=0.36, color=CRITICO, label='Colision')
    ax.set_xticks(x)
    ax.set_xticklabels(['BC nominal' if float(f['perturbacion']) == 0 else 'BC perturbado' for f in datos])
    ax.set_ylim(0, 105)
    ax.set_ylabel('Episodios (%)')
    ax.set_title('Behavior Cloning: evaluacion nominal vs perturbacion inicial')
    ax.legend()
    guardar(fig, salida/'bc_distribution_shift.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corrida', default='', help='corrida de BC y DAgger; vacio usa los resultados originales')
    p.add_argument('--bc-corrida', help='corrida de BC si tiene otro nombre')
    p.add_argument('--salida', default=str(RAIZ/'report'/'B'))
    args = p.parse_args()
    bc_corrida = args.corrida if args.bc_corrida is None else args.bc_corrida
    for nombre in (args.corrida, bc_corrida):
        if nombre and (Path(nombre).name != nombre or nombre in ('.', '..')):
            p.error('Las corridas deben ser nombres de carpeta, sin rutas')
    LOGS = RAIZ/'dagger'/'results'/args.corrida/'logs'
    PESOS = RAIZ/'dagger'/'results'/args.corrida/'weights'
    BC_LOGS = RAIZ/'bc'/'results'/bc_corrida/'logs'
    salida = Path(args.salida)
    if not (LOGS/'rondas.csv').exists():
        raise SystemExit('Falta dagger/results/logs/rondas.csv: ejecuta dagger.py.')
    tasas(salida)
    trayectorias(salida)
    distribution_shift(salida)
