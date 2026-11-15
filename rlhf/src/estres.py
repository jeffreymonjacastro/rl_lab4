"""Parte C - Prueba de estres: tasa de exito en N episodios con ruido gaussiano en el LiDAR.

    python3 rlhf/src/estres.py --episodios 10 --ruidos 0,0.1
Compara pi_DAgger, RLHF con beta>0 y RLHF con beta=0 en los mismos episodios perturbados.
Con --politica se evalua solo una (para repartir entre simuladores); escribe logs/estres_partes/<politica>.csv.
"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
import argparse
import csv
import numpy as np
import rclpy
from base.entorno import Entorno
from base.evaluar import evaluar_archivo
from base.red import dispositivo
from preferencias import DAGGER, PESOS, LOGS, ESCENARIO

SEMILLA = 8000

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodios', type=int, default=10)
    p.add_argument('--ruidos', default='0,0.1', help='desviaciones del ruido LiDAR en metros')
    p.add_argument('--limite', type=int, default=150)
    p.add_argument('--referencia', default=str(DAGGER), help='politica DAgger usada en la Parte C')
    p.add_argument('--politica', choices=['todas', 'dagger', 'beta', 'sin_beta'], default='todas')
    args = p.parse_args()
    claves = {'dagger': ('DAgger', Path(args.referencia)), 'beta': ('RLHF beta>0', PESOS/'beta'/'final.pth'),
              'sin_beta': ('RLHF beta=0', PESOS/'sin_beta'/'final.pth')}
    modelos = list(claves.values()) if args.politica == 'todas' else [claves[args.politica]]
    modelos = [(n, m) for n, m in modelos if m.exists() or print('Falta', m)]
    device = dispositivo()
    filas = []
    rclpy.init()
    env = Entorno(escenario=ESCENARIO, limite=args.limite, perturbacion=1.0, sincronizar=True)
    try:
        env.esperar_sensores()
        for sigma in [float(x) for x in args.ruidos.split(',')]:
            env.ruido_lidar = sigma
            for nombre, pesos in modelos:
                np.random.seed(SEMILLA)
                print(f'--- {nombre}, ruido LiDAR sigma={sigma} m ---', flush=True)
                etiqueta = nombre.lower().replace(' ', '_').replace('>', 'pos').replace('=', '')
                exito, colision = evaluar_archivo(pesos, LOGS/'estres'/f'{etiqueta}_ruido{sigma}', env,
                                                  args.episodios, device, SEMILLA)
                filas.append([nombre, sigma, args.episodios, exito, colision])
    finally:
        try:
            env.liberar()
        finally:
            env.destroy_node()
            rclpy.shutdown()
    destino = LOGS/'estres.csv' if args.politica == 'todas' else LOGS/'estres_partes'/f'{args.politica}.csv'
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['politica', 'ruido_lidar', 'episodios', 'tasa_exito', 'tasa_colision'])
        w.writerows(filas)
    for f in filas:
        print(f'{f[0]:12s} sigma={f[1]:<5} exito={f[3]:.0%} colision={f[4]:.0%}')
