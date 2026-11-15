"""Parte B - DAgger con intervencion humana (compuerta HG-DAgger, Kelly et al. 2019).

Ronda i: pilotea pi_{i-1} (pi_0 = pi_BC) con perturbacion inicial. Mantener W/A/S/D toma el control y al
soltar vuelve la politica; solo se guardan los pares (s, a_experto) de la intervencion: D <- D U D_i.
Luego fine-tuning sobre D (80 epocas, lr 5e-4) y evaluacion autonoma en los mismos 10 episodios.

    python3 dagger/src/dagger.py --rondas 4 --episodios 8        # continua desde la ultima ronda hecha
    python3 dagger/src/dagger.py --rondas 4 --final 3 --mapa 20  # entrega la ronda 3 + mapa de 20 episodios
--final N copia dagger_N.pth -> dagger_sim.pth (por defecto la ultima ronda).
--mapa K evalua bc y dagger_final en K episodios (logs/mapa_*), para la grafica de trayectorias.
"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
import argparse
import shutil
import json
import hashlib
import numpy as np
import rclpy
import torch
from base.imitacion import (ESCENARIO, SEMILLA_EVAL, TRAY_HEADER, Teclado, ajustar, sesion,
                       cargar_datos, guardar_datos, escribir_csv)
from base.entorno import Entorno
from base.evaluar import evaluar_archivo
from base.red import cargar_politica, dispositivo, guardar_politica

RESULTADOS = RAIZ/'dagger'/'results'
DATOS, PESOS, LOGS = RESULTADOS/'data', RESULTADOS/'weights', RESULTADOS/'logs'
BC_RESULTADOS = RAIZ/'bc'/'results'
BC_DATOS, BC_PESOS, BC_LOGS = BC_RESULTADOS/'data', BC_RESULTADOS/'weights', BC_RESULTADOS/'logs'


def configurar_corrida(nombre, bc_nombre=None):
    global RESULTADOS, DATOS, PESOS, LOGS, BC_RESULTADOS, BC_DATOS, BC_PESOS, BC_LOGS
    bc_nombre = nombre if bc_nombre is None else bc_nombre
    for valor in (nombre, bc_nombre):
        if valor and (Path(valor).name != valor or valor in ('.', '..')):
            raise ValueError('--corrida y --bc-corrida deben ser nombres de carpeta, sin rutas.')
    RESULTADOS = RAIZ/'dagger'/'results'/nombre
    BC_RESULTADOS = RAIZ/'bc'/'results'/bc_nombre
    DATOS, PESOS, LOGS = RESULTADOS/'data', RESULTADOS/'weights', RESULTADOS/'logs'
    BC_DATOS, BC_PESOS, BC_LOGS = BC_RESULTADOS/'data', BC_RESULTADOS/'weights', BC_RESULTADOS/'logs'

RONDAS_HEADER = ['ronda', 'piloto', 'pasos', 'pasos_humanos', 'intervenciones', 'episodios', 'colisiones',
                 'exitos', 'pares_agregados', 'pares_dataset']
EVAL_HEADER = ['politica', 'ronda', 'episodios', 'tasa_exito', 'tasa_colision']


def ronda_interactiva(ronda, args, env, teclado, device):
    piloto = BC_PESOS/'bc.pth' if ronda == 1 else PESOS/f'dagger_{ronda-1}.pth'
    red = cargar_politica(piloto, device).eval()
    if not teclado.esperar(' ', f'RONDA {ronda}/{args.rondas}',
                           f'Pilotea {piloto.name}. Deja que conduzca sola.\n'
                           'Manten W/A/S/D SOLO cuando vaya a chocar o se desvie;\n'
                           'suelta para devolverle el control.\n\nESPACIO para empezar'):
        return False
    s, a, tray, st, abortado = sesion(env, teclado, red, args.episodios, device,
                                      f'DAgger ronda {ronda} - pilotea {piloto.name}')
    if abortado:
        print('Ronda abortada: no se guarda nada de esta ronda.')
        return False
    guardar_datos(DATOS/f'd{ronda}.npz', s, a)
    s0, a0 = cargar_datos(DATOS/'agregado.npz')
    s_all = np.concatenate((s0, np.asarray(s, np.float32).reshape(-1, 52)))
    a_all = np.concatenate((a0, np.asarray(a, np.float32).reshape(-1, 2)))
    guardar_datos(DATOS/'agregado.npz', s_all, a_all)
    escribir_csv(LOGS/f'trayectoria_ronda_{ronda}.csv', TRAY_HEADER, tray)
    escribir_csv(LOGS/'rondas.csv', RONDAS_HEADER,
                 [[ronda, piloto.name, st['pasos'], st['pasos_humanos'], st['intervenciones'], st['episodios'],
                   st['colisiones'], st['exitos'], len(s), len(s_all)]], agregar=True)
    print(f'Ronda {ronda}: {st["intervenciones"]} intervenciones, {st["pasos_humanos"]}/{st["pasos"]} pasos humanos, '
          f'{st["colisiones"]} colisiones. Dataset: {len(s_all)} pares', flush=True)
    teclado.estado('AJUSTANDO', '#2a78d6', 'Fine-tuning sobre el dataset agregado...')
    if len(s):
        ajustar(red, s_all, a_all, args.epocas, args.lr, device)
    guardar_politica(PESOS/f'dagger_{ronda}.pth', red, metodo='dagger', ronda=ronda, pares=len(s_all))
    teclado.estado('EVALUANDO', '#2a78d6', f'dagger_{ronda}: {args.eval} episodios sin ayuda.\nNo toques nada.')
    return True


def evaluar_autonomo(nombre, ronda, env, args, device):
    pesos = BC_PESOS if nombre == 'bc' else PESOS
    logs = BC_LOGS if nombre == 'bc' else LOGS
    exito, colision = evaluar_archivo(pesos/f'{nombre}.pth', logs/f'eval_{nombre}', env, args.eval, device, SEMILLA_EVAL)
    escribir_csv(LOGS/'dagger_evaluacion.csv', EVAL_HEADER, [[nombre, ronda, args.eval, exito, colision]], agregar=True)
    print(f'{nombre}: exito={exito:.0%} colision={colision:.0%}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corrida', default='', help='subcarpeta de dagger/results/ y, por defecto, bc/results/')
    p.add_argument('--bc-corrida', help='corrida de BC si tiene otro nombre')
    p.add_argument('--rondas', type=int, default=4)
    p.add_argument('--episodios', type=int, default=8, help='episodios supervisados por ronda')
    p.add_argument('--eval', type=int, default=10, help='episodios autonomos de evaluacion por politica')
    p.add_argument('--limite', type=int, default=150)
    p.add_argument('--epocas', type=int, default=80)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--final', type=int, default=None, help='ronda entregada como dagger_sim.pth (def.: la ultima)')
    p.add_argument('--mapa', type=int, default=0, help='episodios del mapa de trayectorias bc vs dagger_final')
    p.add_argument('--semilla', type=int, default=42)
    args = p.parse_args()
    if min(args.rondas, args.episodios, args.eval, args.limite, args.epocas) < 1 or args.mapa < 0:
        p.error('rondas, episodios, eval, limite y epocas deben ser positivos; mapa no puede ser negativo')
    if args.final is not None and not 1 <= args.final <= args.rondas:
        p.error('--final debe estar entre 1 y --rondas')
    configurar_corrida(args.corrida, args.bc_corrida)
    device = dispositivo()
    np.random.seed(args.semilla)
    torch.manual_seed(args.semilla)
    if not (BC_PESOS/'bc.pth').exists():
        raise SystemExit(f'Falta {BC_PESOS/"bc.pth"}: ejecuta antes bc.py demos y bc.py entrenar con la misma corrida.')
    if not (BC_DATOS/'d0.npz').exists():
        raise SystemExit(f'Falta el dataset inicial: {BC_DATOS/"d0.npz"}')
    LOGS.mkdir(parents=True, exist_ok=True)
    identidad = dict(bc_sha256=hashlib.sha256((BC_PESOS/'bc.pth').read_bytes()).hexdigest(),
                     d0_sha256=hashlib.sha256((BC_DATOS/'d0.npz').read_bytes()).hexdigest())
    configuracion = LOGS/'configuracion.json'
    if configuracion.exists():
        anterior = json.loads(configuracion.read_text())
        if any(anterior.get(k) != v for k, v in identidad.items()):
            raise SystemExit('Los pesos BC o D0 cambiaron desde la primera ronda. Usa una --corrida nueva.')
    else:
        configuracion.write_text(json.dumps(dict(**vars(args), **identidad), indent=2))
    if not (DATOS/'agregado.npz').exists():
        DATOS.mkdir(parents=True, exist_ok=True)
        shutil.copy(BC_DATOS/'d0.npz', DATOS/'agregado.npz')
    hechas = 0
    while (PESOS/f'dagger_{hechas+1}.pth').exists():
        hechas += 1
    rclpy.init()
    env = Entorno(escenario=ESCENARIO, limite=args.limite, perturbacion=1.0)
    env.rng = np.random.default_rng(args.semilla+hechas)
    teclado = Teclado('Lab 4 - DAgger') if hechas < args.rondas else None
    try:
        env.esperar_sensores()
        if not (LOGS/'dagger_evaluacion.csv').exists():
            evaluar_autonomo('bc', 0, env, args, device)
        for ronda in range(hechas+1, args.rondas+1):
            if not ronda_interactiva(ronda, args, env, teclado, device):
                break
            evaluar_autonomo(f'dagger_{ronda}', ronda, env, args, device)
            hechas = ronda
        final = min(args.final or hechas, hechas)
        if final:
            shutil.copy(PESOS/f'dagger_{final}.pth', PESOS/'dagger_sim.pth')
            print(f'Politica entregada: {PESOS/"dagger_sim.pth"} (ronda {final})', flush=True)
            for nombre in (['bc', f'dagger_{final}'] if args.mapa else []):
                print(f'--- mapa: {nombre}, {args.mapa} episodios ---', flush=True)
                pesos = BC_PESOS if nombre == 'bc' else PESOS
                logs = BC_LOGS if nombre == 'bc' else LOGS
                evaluar_archivo(pesos/f'{nombre}.pth', logs/f'mapa_{nombre}', env, args.mapa, device, SEMILLA_EVAL)
    finally:
        env.parar()
        if teclado:
            teclado.cerrar()
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
