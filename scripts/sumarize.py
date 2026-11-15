#!/usr/bin/env python3
"""Reune en informe/ las graficas, pesos y tablas a entregar de las Partes A, B y C.

    python3 scripts/sumarize.py [--escenario-a cruce|facil]
Regenera las graficas desde los logs y copia lo que exista; avisa de lo que falte.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
INFORME = RAIZ/'informe'
faltan = []


def copiar(origen, destino):
    origen = Path(origen)
    if not origen.exists():
        faltan.append(str(origen.relative_to(RAIZ)))
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(origen, destino)
    print('  +', destino.relative_to(RAIZ))


def graficar(script, salida, *extra):
    r = subprocess.run([sys.executable, str(RAIZ/script), '--salida', str(salida), *extra],
                       capture_output=True, text=True)
    if r.returncode:
        faltan.append(f'{script}: {(r.stderr or r.stdout).strip().splitlines()[-1]}')
    for linea in r.stdout.splitlines():
        if linea.startswith('Falta') or linea.startswith('Aun'):
            faltan.append(f'{script}: {linea}')


def parte_a(escenario):
    print(f'Parte A (escenario {escenario})')
    d = INFORME/'parte_a'
    graficar('scripts/graficar_a.py', d/'graficas', '--corrida', escenario)
    mejor = None
    for metodo in ('a2c', 'ppo'):
        res = RAIZ/metodo/'results'
        copiar(res/'weights'/escenario/'actor_base.pth', d/'pesos'/f'actor_base_{metodo}.pth')
        for nombre in ('episodios.csv', 'validacion.csv', 'configuracion.json', 'eval_episodios.csv'):
            copiar(res/'logs'/escenario/nombre, d/'datos'/f'{metodo}_{nombre}')
        val = res/'logs'/escenario/'validacion.csv'
        if val.exists():
            import csv
            por_paso = {}
            for f in csv.DictReader(open(val)):
                por_paso.setdefault(f['pasos_entrenamiento'], []).append((f['exito'] == 'True', float(f['recompensa'])))
            score = max((sum(e for e, _ in v)/len(v), sum(r for _, r in v)/len(v)) for v in por_paso.values())
            if mejor is None or score > mejor[0]:
                mejor = (score, metodo)
    if mejor:  # actor_base.pth = mejor validacion entre A2C y PPO
        copiar(RAIZ/mejor[1]/'results'/'weights'/escenario/'actor_base.pth', d/'pesos'/'actor_base.pth')
        (d/'pesos'/'LEEME.txt').write_text(
            f'actor_base.pth es el de {mejor[1].upper()} (exito de validacion {mejor[0][0]:.0%}, '
            f'recompensa media {mejor[0][1]:.1f}), escenario {escenario}.\n')


def parte_b():
    print('Parte B')
    d = INFORME/'parte_b'
    res = RAIZ/'dagger'/'results'
    bc = RAIZ/'bc'/'results'
    graficar('scripts/graficar_b.py', d/'graficas')
    copiar(bc/'weights'/'bc.pth', d/'pesos'/'bc.pth')
    copiar(bc/'logs'/'bc_distribution_shift.csv', d/'datos'/'bc_distribution_shift.csv')
    for archivo in sorted((bc/'logs').glob('mapa_*_episodios.csv')):
        copiar(archivo, d/'datos'/archivo.name)
    for nombre in ['dagger_sim.pth']+sorted(p.name for p in (res/'weights').glob('dagger_[0-9]*.pth')):
        copiar(res/'weights'/nombre, d/'pesos'/nombre)
    for nombre in ['rondas.csv', 'dagger_evaluacion.csv']+sorted(
            p.name for p in (res/'logs').glob('mapa_*_episodios.csv')):
        copiar(res/'logs'/nombre, d/'datos'/nombre)


def parte_c():
    print('Parte C')
    d = INFORME/'parte_c'
    res = RAIZ/'rlhf'/'results'
    graficar('scripts/graficar_c.py', d/'graficas')
    copiar(res/'weights'/'recompensa_bt.pth', d/'pesos'/'recompensa_bt.pth')
    copiar(res/'weights'/'beta'/'final.pth', d/'pesos'/'rlhf_beta.pth')
    copiar(res/'weights'/'sin_beta'/'final.pth', d/'pesos'/'rlhf_sin_beta.pth')
    copiar(res/'data'/'preferencias.csv', d/'datos'/'preferencias.csv')
    for nombre in ('bt_perdida.csv', 'estres.csv'):
        copiar(res/'logs'/nombre, d/'datos'/nombre)
    for corrida in ('beta', 'sin_beta'):
        for nombre in ('validacion.csv', 'episodios.csv', 'configuracion.json', 'kl_referencia.csv'):
            copiar(res/'logs'/corrida/nombre, d/'datos'/f'rlhf_{corrida}_{nombre}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--escenario-a', choices=['cruce', 'facil'], default='cruce')
    args = p.parse_args()
    if INFORME.exists():
        shutil.rmtree(INFORME)
    parte_a(args.escenario_a)
    parte_b()
    parte_c()
    print('\nListo:', INFORME)
    if faltan:
        print('\nPENDIENTE (aun no existe):')
        for f in faltan:
            print('  -', f)
