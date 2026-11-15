"""Evaluacion de una politica guardada: N episodios con la accion media (sin ruido) y semillas fijas.

    python3 base/evaluar.py --pesos X.pth --salida carpeta/prefijo [--episodios 10] [--perturbacion 1]
                             [--ruido-lidar 0.1] [--tiempo-libre]
Escribe <salida>_episodios.csv y <salida>_pasos.csv. Con --tiempo-libre Gazebo corre en tiempo real (video).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import numpy as np
import torch

PASO_HEADER = ['episodio', 'paso', 'v', 'omega', 'x', 'y', 'tiempo_sim', 'reward', 'distancia_meta', 'distancia_min']
EVAL_HEADER = ['episodio', 'recompensa', 'pasos', 'exito', 'colision']
SEMILLA_VALIDACION = 5000


def fila_paso(ep, k, info, r):
    return [ep, k, info['v'], info['omega'], info['x'], info['y'], info['tiempo_sim'], r,
            info['distancia_meta'], info['distancia_min']]


def csv_append(path, header, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(header)
        w.writerows(rows)


def evaluar(red, env, n, device, semilla_base=SEMILLA_VALIDACION):
    """Episodio ep usa reset(semilla_base+ep): mismos escenarios para cualquier politica."""
    resultados, pasos = [], []
    red.eval()
    for ep in range(1, n+1):
        obs = env.reset(semilla_base+ep)
        total = 0.
        for k in range(1, env.limite+1):
            with torch.no_grad():
                a = red.action(torch.as_tensor(obs, device=device), True)[0].cpu().numpy()
            obs, r, done, trunc, info = env.step(a)
            total += r
            pasos.append(fila_paso(ep, k, info, r))
            if done or trunc:
                break
        resultados.append([ep, total, k, info['exito'], info['colision']])
        print(f'Prueba {ep}/{n}: recompensa={total:.2f}, pasos={k}, exito={info["exito"]}, '
              f'colision={info["colision"]}', flush=True)
    red.train()
    return resultados, pasos


def evaluar_archivo(pesos, salida, env, episodios, device, semilla_base=SEMILLA_VALIDACION):
    """Evalua unos pesos, guarda los CSV y devuelve (tasa de exito, tasa de colision)."""
    from base.red import cargar_politica
    filas, pasos = evaluar(cargar_politica(pesos, device), env, episodios, device, semilla_base)
    for sufijo, header, datos in [('_episodios.csv', EVAL_HEADER, filas), ('_pasos.csv', PASO_HEADER, pasos)]:
        Path(str(salida)+sufijo).unlink(missing_ok=True)
        csv_append(str(salida)+sufijo, header, datos)
    return sum(f[3] for f in filas)/len(filas), sum(f[4] for f in filas)/len(filas)


def main():
    import rclpy
    from base.entorno import Entorno, ESCENARIOS
    from base.red import dispositivo
    p = argparse.ArgumentParser()
    p.add_argument('--pesos', required=True)
    p.add_argument('--salida', required=True)
    p.add_argument('--escenario', choices=list(ESCENARIOS), default='cruce')
    p.add_argument('--episodios', type=int, default=10)
    p.add_argument('--limite', type=int, default=150)
    p.add_argument('--perturbacion', type=float, default=0.)
    p.add_argument('--ruido-lidar', type=float, default=0.)
    p.add_argument('--semilla', type=int, default=SEMILLA_VALIDACION)
    p.add_argument('--tiempo-libre', action='store_true')
    args = p.parse_args()
    np.random.seed(args.semilla)
    torch.manual_seed(args.semilla)
    rclpy.init()
    env = Entorno(escenario=args.escenario, limite=args.limite, ruido_lidar=args.ruido_lidar,
                  sincronizar=not args.tiempo_libre, perturbacion=args.perturbacion)
    try:
        env.esperar_sensores()
        exito, colision = evaluar_archivo(args.pesos, args.salida, env, args.episodios, dispositivo(), args.semilla)
        print(f'Tasa de exito={exito:.0%} | tasa de colision={colision:.0%}')
    finally:
        env.liberar()
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
