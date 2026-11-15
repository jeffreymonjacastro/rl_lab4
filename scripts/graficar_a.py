"""Parte A - Graficas: (1) recompensa acumulada vs episodios y (2) perfil v(t), w(t) de un episodio de prueba.

    python3 scripts/graficar_a.py --corrida cruce [--salida carpeta]
"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import argparse
import numpy as np
from graficas import plt, filas, guardar, AZUL, NARANJA, TENUE

METODOS = [('a2c', 'A2C', AZUL), ('ppo', 'PPO', NARANJA)]


def logs(metodo, corrida):
    return RAIZ/metodo/'results'/'logs'/corrida


def episodio_prueba(carpeta):
    """Pasos de un episodio de prueba: evaluacion final si existe; si no, la ultima validacion."""
    if (carpeta/'eval_pasos.csv').exists():
        datos = filas(carpeta/'eval_pasos.csv')
        exitos = [f['episodio'] for f in filas(carpeta/'eval_episodios.csv') if f['exito'] == 'True']
        ep = exitos[0] if exitos else datos[0]['episodio']
        return [f for f in datos if f['episodio'] == ep]
    datos = filas(carpeta/'validacion_pasos.csv')
    ultimo = max(int(f['pasos_entrenamiento']) for f in datos)
    return [f for f in datos if int(f['pasos_entrenamiento']) == ultimo and f['episodio'] == '1']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corrida', default='cruce')
    p.add_argument('--corrida-a2c', help='corrida de A2C si tiene un nombre distinto')
    p.add_argument('--corrida-ppo', help='corrida de PPO si tiene un nombre distinto')
    p.add_argument('--solo-recompensa', action='store_true', help='omite el perfil de velocidades')
    p.add_argument('--salida', default='')
    p.add_argument('--ventana', type=int, default=20)
    args = p.parse_args()
    if args.ventana < 1:
        p.error('--ventana debe ser mayor que cero')
    corridas = {'a2c': args.corrida_a2c or args.corrida, 'ppo': args.corrida_ppo or args.corrida}
    salida = Path(args.salida) if args.salida else RAIZ/'report'/'parte_a'/'graficas'/args.corrida
    presentes = [(m, n, c) for m, n, c in METODOS if (logs(m, corridas[m])/'episodios.csv').exists()]
    if not presentes:
        raise SystemExit(f'No hay corridas "{args.corrida}" en a2c/results/logs ni ppo/results/logs.')

    fig, ax = plt.subplots(figsize=(8, 4.2))
    dibujadas = 0
    for metodo, nombre, color in presentes:
        datos = [f for f in filas(logs(metodo, corridas[metodo])/'episodios.csv') if f['interrumpido'] != 'True']
        if not datos:
            print(f'Sin episodios completos: {metodo}/{corridas[metodo]}; se omite.')
            continue
        r = np.array([float(f['recompensa']) for f in datos])
        episodios = np.array([int(f['episodio']) for f in datos])
        ax.plot(episodios, r, color=color, alpha=0.35, linewidth=1,
                marker='o' if len(r) <= 20 else None, markersize=3)
        k = min(args.ventana, len(r))
        media = np.convolve(r, np.ones(k)/k, mode='valid')
        desv = np.array([r[i:i+k].std() for i in range(len(media))])
        x = episodios[k-1:]
        ax.plot(x, media, color=color, linestyle='-' if metodo == 'a2c' else '--',
                marker='o' if len(media) <= 20 else None,
                label=f'{nombre} (media movil de {k})')
        ax.fill_between(x, media-desv, media+desv, color=color, alpha=0.12, linewidth=0)
        dibujadas += 1
        print(f'{nombre}: {len(datos)} episodios completos; fuente: {logs(metodo, corridas[metodo])}')
    if not dibujadas:
        plt.close(fig)
        raise SystemExit('No hay episodios completos para graficar.')
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa acumulada')
    ax.set_title(f'Parte A: A2C vs PPO, escenario "{args.corrida}" (banda = ±1 desv. movil)')
    ax.legend()
    guardar(fig, salida/'recompensa_a2c_vs_ppo.png')
    if args.solo_recompensa:
        return

    fig, ejes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    for metodo, nombre, color in presentes:
        datos = episodio_prueba(logs(metodo, corridas[metodo]))
        t = np.array([float(f['tiempo_sim']) for f in datos])
        t = t-t[0]+0.2
        w = np.array([float(f['omega']) for f in datos])
        ejes[0].step(t, [float(f['v']) for f in datos], where='post', color=color, label=nombre)
        ejes[1].step(t, w, where='post', color=color,
                     label=f'{nombre}: media |Δω| = {np.abs(np.diff(w)).mean() if len(w) > 1 else 0:.2f} rad/s')
    ejes[0].axhline(0.22, color=TENUE, linestyle=':', linewidth=1)
    ejes[0].set_ylabel('v (m/s)')
    ejes[0].set_ylim(0, 0.24)
    ejes[1].set_ylabel('ω (rad/s)')
    ejes[1].set_xlabel('Tiempo de simulacion (s)')
    ejes[0].set_title(f'Parte A: perfil de velocidades en un episodio de prueba ("{args.corrida}")')
    ejes[0].legend()
    ejes[1].legend()
    guardar(fig, salida/'velocidades_a2c_vs_ppo.png')


if __name__ == '__main__':
    main()
