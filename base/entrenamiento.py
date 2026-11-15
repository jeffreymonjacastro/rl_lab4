"""Bucle on-policy comun a A2C, PPO (Parte A) y RLHF (Parte C).

Cada algoritmo aporta solo `actualizar(red, opt, lote, args) -> metricas`. Este modulo hace los rollouts,
calcula GAE, valida cada `--validar-cada` pasos y guarda:
  <resultados>/weights/<corrida>/{actor_base, final}.pth   (actor_base = mejor validacion)
  <resultados>/logs/<corrida>/{configuracion.json, episodios.csv, pasos.csv, validacion*.csv, actualizaciones.csv}
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import rclpy
import torch
from base.entorno import Entorno, ESCENARIOS
from base.evaluar import PASO_HEADER, csv_append, evaluar, fila_paso
from base.red import ActorCritico, dispositivo, guardar_politica

EP_HEADER = ['episodio', 'recompensa', 'pasos', 'exito', 'colision', 'interrumpido', 'recompensa_entorno']
VAL_HEADER = ['pasos_entrenamiento', 'prueba', 'recompensa', 'pasos', 'exito', 'colision']
ACT_HEADER = ['actualizacion', 'pasos', 'actor_loss', 'critic_loss', 'entropia', 'kl', 'clip_fraction', 'grad_norm']


def gae(rewards, values, next_values, fin, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation. next_values ya vale 0 en estados terminales (no en truncados)."""
    adv = torch.zeros_like(rewards)
    acumulada = torch.zeros((), device=rewards.device)
    for t in reversed(range(len(rewards))):
        delta = rewards[t]+gamma*next_values[t]-values[t]
        acumulada = delta+gamma*lam*(1-fin[t])*acumulada
        adv[t] = acumulada
    return adv, adv+values


def preparar_lote(buffer, args, device):
    """Rollout -> tensores con ventaja GAE normalizada y retornos."""
    ss, zz, old = (torch.stack([x[i] for x in buffer]) for i in range(3))
    rr, vv, nv, ff = (torch.tensor([x[i] for x in buffer], device=device, dtype=torch.float32) for i in range(3, 7))
    adv, ret = gae(rr, vv, nv, ff, args.gamma, args.gae_lambda)
    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    return dict(s=ss, z=zz, logp=old.detach(), adv=adv, ret=ret)


def parser_base(descripcion):
    p = argparse.ArgumentParser(description=descripcion)
    p.add_argument('--corrida', required=True, help='subcarpeta de weights/ y logs/')
    p.add_argument('--escenario', choices=list(ESCENARIOS), default='cruce')
    p.add_argument('--pasos', type=int, default=10000)
    p.add_argument('--episodios', type=int,
                   help='episodios completos a entrenar; si se usa, reemplaza --pasos y valida entre episodios')
    p.add_argument('--limite', type=int, default=150, help='pasos maximos por episodio')
    p.add_argument('--rollout', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--entropia', type=float, default=0.002)
    p.add_argument('--coef-critico', type=float, default=0.5)
    p.add_argument('--escala-recompensa', type=float, default=0.1)
    p.add_argument('--sesgo-avance', type=float, default=0.35, help='sesgo inicial del actor para v')
    p.add_argument('--log-std-inicial', type=float, default=-1.0)
    p.add_argument('--validar-cada', type=int, default=1000)
    p.add_argument('--pruebas-validacion', type=int, default=3)
    p.add_argument('--ruido-lidar', type=float, default=0.)
    p.add_argument('--perturbacion', type=float, default=0., help='escala de la perturbacion inicial')
    p.add_argument('--semilla', type=int, default=42)
    return p


def entrenar(args, metodo, actualizar, resultados, red=None, recompensa_extra=None):
    """red: politica inicial (RLHF) o None (desde cero).
    recompensa_extra(s, a, z, logp, r_entorno): reemplaza la recompensa de entrenamiento (RLHF)."""
    device = dispositivo()
    if args.episodios is not None and args.episodios < 1:
        raise SystemExit('--episodios debe ser mayor que cero.')
    if args.episodios is not None and args.limite < 1:
        raise SystemExit('--limite debe ser mayor que cero cuando se usa --episodios.')
    pasos_objetivo = args.pasos if args.episodios is None else args.episodios*args.limite
    torch.set_num_threads(2)
    torch.manual_seed(args.semilla)
    np.random.seed(args.semilla)
    pesos, logs = Path(resultados)/'weights'/args.corrida, Path(resultados)/'logs'/args.corrida
    if (logs/'configuracion.json').exists():
        raise SystemExit(f'La corrida {args.corrida} ya existe en {logs}; usa otro --corrida o borrala.')
    logs.mkdir(parents=True, exist_ok=True)
    (logs/'configuracion.json').write_text(json.dumps(dict(**vars(args), metodo=metodo, device=device), indent=2))
    if red is None:
        red = ActorCritico().to(device)
        with torch.no_grad():
            red.actor[-1].bias[0] = args.sesgo_avance
            red.log_std.fill_(args.log_std_inicial)
    opt = torch.optim.Adam(red.parameters(), lr=args.lr, eps=1e-5)

    def guardar(nombre, pasos, score=None):
        guardar_politica(pesos/nombre, red, metodo=metodo, pasos=pasos, configuracion=vars(args),
                         score_validacion=score)

    def validar(pasos):
        rows, steps = evaluar(red, env, args.pruebas_validacion, device)
        csv_append(logs/'validacion.csv', VAL_HEADER, [[pasos]+r for r in rows])
        csv_append(logs/'validacion_pasos.csv', ['pasos_entrenamiento']+PASO_HEADER, [[pasos]+r for r in steps])
        return sum(r[3] for r in rows)/len(rows), float(np.mean([r[1] for r in rows]))  # (exito, recompensa)

    rclpy.init()
    env = Entorno(escenario=args.escenario, limite=args.limite, ruido_lidar=args.ruido_lidar,
                  sincronizar=True, perturbacion=args.perturbacion)
    env.rng = np.random.default_rng(args.semilla)
    paso, ep, ep_pasos, total, total_env, n_act = 0, 1, 0, 0., 0., 0
    buffer = []
    siguiente_validacion = args.validar_cada
    inicio = time.monotonic()
    try:
        env.esperar_sensores()
        if env.count_publishers('/cmd_vel') > 1:
            raise RuntimeError('Hay otro controlador en /cmd_vel; detenlo antes de entrenar.')
        objetivo = (f'{args.pasos} pasos' if args.episodios is None else
                    f'{args.episodios} episodios (maximo {pasos_objetivo} pasos)')
        print(f'{metodo.upper()} | escenario={args.escenario} | dispositivo={device} | {objetivo}', flush=True)
        print('VALIDACION INICIAL (antes de aprender)', flush=True)
        mejor = validar(0)
        guardar('actor_base.pth', 0, mejor)
        obs = env.reset()
        while paso < pasos_objetivo and (args.episodios is None or ep <= args.episodios):
            s = torch.as_tensor(obs, device=device)
            with torch.no_grad():
                a, z, lp, v = red.action(s)
            nxt, r, done, trunc, info = env.step(a.cpu().numpy())
            reward = r if recompensa_extra is None else recompensa_extra(s, a, z, lp, r)
            with torch.no_grad():
                nextv = 0. if done else float(red.critic(torch.as_tensor(nxt, device=device)).item())
            buffer.append((s, z, lp, reward*args.escala_recompensa, float(v), nextv, float(done or trunc)))
            paso += 1
            ep_pasos += 1
            total += reward
            total_env += r
            csv_append(logs/'pasos.csv', PASO_HEADER, [fila_paso(ep, ep_pasos, info, reward)])
            obs = nxt
            if done or trunc:
                csv_append(logs/'episodios.csv', EP_HEADER,
                           [[ep, total, ep_pasos, info['exito'], info['colision'], False, total_env]])
                print(f'Episodio {ep}: pasos={ep_pasos}, recompensa={total:.2f}, exito={info["exito"]}, '
                      f'colision={info["colision"]}, total_pasos={paso}', flush=True)
                ep, ep_pasos, total, total_env = ep+1, 0, 0., 0.
                obs = env.reset()
            ultimo = (paso >= pasos_objetivo or
                      (args.episodios is not None and ep > args.episodios))
            validar_ahora = paso >= siguiente_validacion and (args.episodios is None or ep_pasos == 0)
            if len(buffer) >= args.rollout or validar_ahora or ultimo:
                metricas = actualizar(red, opt, preparar_lote(buffer, args, device), args)
                buffer.clear()
                n_act += 1
                csv_append(logs/'actualizaciones.csv', ACT_HEADER, [[n_act, paso]+metricas])
                avance = (f'{paso}/{pasos_objetivo} pasos' if args.episodios is None else
                          f'{min(ep-1, args.episodios)}/{args.episodios} episodios | {paso}/{pasos_objetivo} pasos')
                print(f'AVANCE {avance} | kl={metricas[3]:.4f} | d_meta={info["distancia_meta"]:.2f}m | '
                      f'min={(time.monotonic()-inicio)/60:.1f}', flush=True)
                if validar_ahora or ultimo:
                    if ep_pasos:  # el episodio en curso se corta para validar
                        csv_append(logs/'episodios.csv', EP_HEADER, [[ep, total, ep_pasos, False, False, True, total_env]])
                        ep += 1
                    print(f'VALIDACION en {paso} pasos', flush=True)
                    score = validar(paso)
                    if score > mejor:
                        mejor = score
                        guardar('actor_base.pth', paso, score)
                    siguiente_validacion = (paso//args.validar_cada+1)*args.validar_cada
                    ep_pasos, total, total_env = 0, 0., 0.
                    obs = env.reset()
        guardar('final.pth', paso)
        print('Finalizado. Pesos en', pesos, flush=True)
    except KeyboardInterrupt:
        guardar('interrumpido.pth', paso)
        print(f'Interrumpido en {paso} pasos; pesos guardados.', flush=True)
    finally:
        env.liberar()
        env.destroy_node()
        rclpy.shutdown()
    return red
