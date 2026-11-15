"""Parte B - Behavior Cloning (linea base).

    python3 bc/src/bc.py demos --episodios 10   # D0: conduces tu (WASD) hasta la meta; choques se descartan
    python3 bc/src/bc.py entrenar               # pi_BC: regresion supervisada sobre D0 -> weights/bc.pth
    python3 bc/src/bc.py evaluar                # 10 episodios nominales vs perturbados (Distribution Shift)
Red: base/red.py (solo el actor). Ajuste: imitacion.ajustar (MSE, Adam 1e-3, 200 epocas).
"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
import argparse
import json
import hashlib
import numpy as np
import rclpy
import torch
from base.imitacion import (ESCENARIO, SEMILLA_EVAL, TRAY_HEADER, Teclado, ajustar, sesion,
                       cargar_datos, guardar_datos, escribir_csv)
from base.entorno import Entorno
from base.evaluar import evaluar_archivo
from base.red import ActorCritico, dispositivo, guardar_politica

RESULTADOS = RAIZ/'bc'/'results'
DATOS, PESOS, LOGS = RESULTADOS/'data', RESULTADOS/'weights', RESULTADOS/'logs'


def configurar_corrida(nombre):
    global RESULTADOS, DATOS, PESOS, LOGS
    if nombre:
        if Path(nombre).name != nombre or nombre in ('.', '..'):
            raise ValueError('--corrida debe ser un nombre de carpeta, sin rutas.')
        RESULTADOS = RAIZ/'bc'/'results'/nombre
    DATOS, PESOS, LOGS = RESULTADOS/'data', RESULTADOS/'weights', RESULTADOS/'logs'


def experto_dqn(args, device):
    from base.experto_dqn import ExpertoDQN
    experto = ExpertoDQN(args.modelo_dqn)
    if args.modo == 'experto':
        q = experto.valores_q(np.r_[1.5, 0., np.full(24, 3.5)])
        print('Experto cargado:', experto.archivo)
        print('Entrada: 26 valores. Salida: 5 valores Q. Inferencia en CPU con PyTorch.')
        print('Comprobacion numerica:', q)
        print('Esta comprobacion no mide su tasa de exito en obstaculos moviles.')
        return
    if (DATOS/'d0.npz').exists() and not args.agregar:
        raise SystemExit(f'Ya existe {DATOS/"d0.npz"}; usa una --corrida nueva o --agregar.')
    estados, acciones, trayectoria, intentos = [], [], [], []
    exitos = 0
    max_intentos = args.intentos or 3*args.episodios
    rclpy.init()
    env = Entorno(escenario=ESCENARIO, limite=args.limite, sincronizar=True)
    try:
        env.esperar_sensores()
        if env.count_publishers('/cmd_vel') > 1:
            raise RuntimeError('Hay otro controlador publicando en /cmd_vel. Detenlo antes de recolectar D0.')
        for intento in range(1, max_intentos+1):
            obs = env.reset(args.semilla+intento)
            ss, aa, tray = [], [], []
            for paso in range(1, args.limite+1):
                accion = experto.accion(env.observar_dqn())
                ss.append(obs.copy())
                aa.append(accion.copy())
                obs, _, done, trunc, info = env.step(accion)
                tray.append([intento, paso, info['x'], info['y'], False, info['v'], info['omega']])
                if done or trunc:
                    break
            intentos.append([intento, paso, info['exito'], info['colision']])
            trayectoria.extend(f+[info['exito'], info['colision']] for f in tray)
            if info['exito']:
                estados.extend(ss)
                acciones.extend(aa)
                exitos += 1
            print(f'DQN intento {intento}/{max_intentos}: exito={info["exito"]}, '
                  f'colision={info["colision"]}; demos aceptadas={exitos}/{args.episodios}', flush=True)
            if exitos >= args.episodios:
                break
    except KeyboardInterrupt:
        print('Recoleccion interrumpida: se conservan las demostraciones completas exitosas.')
    finally:
        try:
            env.liberar()
        finally:
            env.destroy_node()
            rclpy.shutdown()
    escribir_csv(LOGS/'experto_dqn_episodios.csv', ['intento', 'pasos', 'exito', 'colision'], intentos,
                 agregar=args.agregar)
    escribir_csv(LOGS/'trayectoria_experto_dqn.csv', TRAY_HEADER, trayectoria, agregar=args.agregar)
    LOGS.mkdir(parents=True, exist_ok=True)
    metadata = dict(experto='DQN externo', modelo=str(experto.archivo),
                    sha256=hashlib.sha256(experto.archivo.read_bytes()).hexdigest(), escenario=ESCENARIO,
                    intentos=len(intentos), exitos=exitos, pares_aceptados=len(estados),
                    criterio='solo episodios completos con exito', semilla=args.semilla)
    with (LOGS/'experto_dqn_sesiones.jsonl').open('a') as f:
        f.write(json.dumps(metadata)+'\n')
    if not estados:
        raise SystemExit('El DQN no completo demostraciones exitosas: D0 no se modifico. '
                         'Consulta experto_dqn_episodios.csv y usa --experto teclado para recolectar D0.')
    if args.agregar and (DATOS/'d0.npz').exists():
        s0, a0 = cargar_datos(DATOS/'d0.npz')
        estados, acciones = list(s0)+estados, list(a0)+acciones
    guardar_datos(DATOS/'d0.npz', estados, acciones)
    print(f'D0 guardado en {DATOS/"d0.npz"}: {len(estados)} pares; {exitos} demos nuevas exitosas.')


def demos(args, device):
    if args.experto == 'dqn':
        return experto_dqn(args, device)
    if (DATOS/'d0.npz').exists() and not args.agregar:
        raise SystemExit('Ya existe data/d0.npz. Usa --agregar para sumar demostraciones o borralo.')
    rclpy.init()
    env = Entorno(escenario=ESCENARIO, limite=args.limite)
    teclado = Teclado('Lab 4 - demostraciones D0')
    try:
        env.esperar_sensores()
        if not teclado.esperar(' ', 'LISTO', 'Conduce el robot hasta la meta (poste verde) esquivando\n'
                               'los cilindros rojos. Los choques se descartan.\n\nESPACIO para empezar'):
            return
        s, a, tray, st, _ = sesion(env, teclado, None, args.episodios, device, 'Demostraciones D0', solo_exitos=True)
    finally:
        env.parar()
        teclado.cerrar()
        env.destroy_node()
        rclpy.shutdown()
    if not s:
        raise SystemExit('No se grabo ninguna demostracion.')
    if args.agregar and (DATOS/'d0.npz').exists():
        s0, a0 = cargar_datos(DATOS/'d0.npz')
        s, a = list(s0)+s, list(a0)+a
    guardar_datos(DATOS/'d0.npz', s, a)
    escribir_csv(LOGS/'trayectoria_demos.csv', TRAY_HEADER, tray, agregar=args.agregar)
    print(f'D0: {len(s)} pares en {DATOS/"d0.npz"} ({st["episodios"]} episodios)')


def entrenar(args, device):
    if (PESOS/'bc.pth').exists():
        raise SystemExit(f'Ya existe {PESOS/"bc.pth"}; usa una --corrida nueva para no sobrescribirlo.')
    s, a = cargar_datos(DATOS/'d0.npz')
    if not len(s):
        raise SystemExit('D0 esta vacio: recolecta demostraciones antes de entrenar.')
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS/'configuracion.json').write_text(json.dumps(dict(**vars(args),
        d0_sha256=hashlib.sha256((DATOS/'d0.npz').read_bytes()).hexdigest()), indent=2))
    torch.manual_seed(args.semilla)
    red = ActorCritico().to(device)
    historial = ajustar(red, s, a, args.epocas, 1e-3, device)
    guardar_politica(PESOS/'bc.pth', red, metodo='bc', pares=len(s))
    escribir_csv(LOGS/'bc_perdida.csv', ['epoca', 'mse'], historial)
    print('Guardado', PESOS/'bc.pth')


def evaluar(args, device):
    rclpy.init()
    env = Entorno(escenario=ESCENARIO, limite=args.limite)
    filas = []
    try:
        env.esperar_sensores()
        for nombre, perturbacion in [('bc_nominal', 0.), ('bc', 1.)]:
            env.perturbacion = perturbacion
            print(f'--- pi_BC, perturbacion inicial = {perturbacion} ---', flush=True)
            exito, colision = evaluar_archivo(PESOS/'bc.pth', LOGS/f'eval_{nombre}', env, args.episodios,
                                              device, SEMILLA_EVAL)
            filas.append([nombre, perturbacion, args.episodios, exito, colision])
    finally:
        env.parar()
        env.destroy_node()
        rclpy.shutdown()
    escribir_csv(LOGS/'bc_distribution_shift.csv',
                 ['evaluacion', 'perturbacion', 'episodios', 'tasa_exito', 'tasa_colision'], filas)
    for f in filas:
        print(f'{f[0]:11s} exito={f[3]:.0%} colision={f[4]:.0%}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    from base.experto_dqn import MODELO_DQN
    p.add_argument('modo', choices=['demos', 'entrenar', 'evaluar', 'experto'])
    p.add_argument('--corrida', default='', help='subcarpeta nueva dentro de bc/results/')
    p.add_argument('--experto', choices=['teclado', 'dqn'], default='teclado')
    p.add_argument('--modelo-dqn', default=str(MODELO_DQN))
    p.add_argument('--intentos', type=int, help='maximo de intentos DQN; por defecto 3 por demo solicitada')
    p.add_argument('--episodios', type=int, default=10)
    p.add_argument('--limite', type=int, default=150)
    p.add_argument('--epocas', type=int, default=200)
    p.add_argument('--semilla', type=int, default=42)
    p.add_argument('--agregar', action='store_true')
    args = p.parse_args()
    if min(args.episodios, args.limite, args.epocas) < 1 or (args.intentos is not None and args.intentos < 1):
        p.error('episodios, limite, epocas e intentos deben ser positivos')
    configurar_corrida(args.corrida)
    np.random.seed(args.semilla)
    {'demos': demos, 'entrenar': entrenar, 'evaluar': evaluar, 'experto': experto_dqn}[args.modo](args, dispositivo())
