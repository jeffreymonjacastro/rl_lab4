"""Parte C - Grafica (5) perdida Bradley-Terry y tabla de tasa de exito bajo ruido LiDAR.

    python3 scripts/graficar_c.py [--salida carpeta]
"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import argparse
from graficas import plt, filas, guardar, AZUL

LOGS = RAIZ/'rlhf'/'results'/'logs'


def perdida(salida):
    datos = filas(LOGS/'bt_perdida.csv')
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot([int(f['epoca']) for f in datos], [float(f['perdida']) for f in datos], color=AZUL)
    ax.set_xlabel('Epoca')
    ax.set_ylabel('L_RM = −ln σ(R(τw) − R(τl))')
    ax.set_title(f'Perdida del modelo de recompensa Bradley-Terry (acierto final '
                 f'{float(datos[-1]["acierto"]):.0%} de los pares)')
    ax.set_ylim(bottom=0)
    guardar(fig, salida/'perdida_bradley_terry.png')


def tabla(salida):
    datos = filas(LOGS/'estres.csv')
    ruidos = sorted({float(f['ruido_lidar']) for f in datos})
    politicas = list(dict.fromkeys(f['politica'] for f in datos))
    n = datos[0]['episodios']
    celdas = {(f['politica'], float(f['ruido_lidar'])): f for f in datos}
    lineas = [f'Tasa de exito en {n} episodios (colisiones entre parentesis), escenario de cruce con '
              'perturbacion inicial.', '',
              '| Politica | '+' | '.join(f'Ruido LiDAR σ = {r:g} m' for r in ruidos)+' |',
              '|---|'+'---|'*len(ruidos)]
    for pol in politicas:
        fila = [f'{float(celdas[pol, r]["tasa_exito"]):.0%} ({float(celdas[pol, r]["tasa_colision"]):.0%})'
                if (pol, r) in celdas else '-' for r in ruidos]
        lineas.append(f'| {pol} | '+' | '.join(fila)+' |')
    salida.mkdir(parents=True, exist_ok=True)
    (salida/'tabla_estres.md').write_text('\n'.join(lineas)+'\n')
    print('\n'.join(lineas))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--salida', default=str(RAIZ/'rlhf'/'results'/'graphs'))
    salida = Path(p.parse_args().salida)
    if (LOGS/'bt_perdida.csv').exists():
        perdida(salida)
    else:
        print('Falta bt_perdida.csv: ejecuta preferencias.py entrenar.')
    if (LOGS/'estres.csv').exists():
        tabla(salida)
    else:
        print('Falta estres.csv: ejecuta estres.py.')
