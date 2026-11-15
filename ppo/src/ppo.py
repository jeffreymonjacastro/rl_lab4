"""Parte A - Proximal Policy Optimization (PPO) con objetivo recortado.

    r_t(theta) = pi_theta(a|s) / pi_old(a|s)
    L_CLIP = E[min(r_t * A, clip(r_t, 1-eps, 1+eps) * A)],  eps = 0.2
4 epocas por rollout en minibatches de 64, con parada temprana si KL > 0.03.

    python3 ppo/src/ppo.py --corrida cruce --escenario cruce --pasos 20000
Salida: ppo/results/{weights,logs}/<corrida>/.  La Parte C reutiliza `actualizar_ppo`.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import nn
from base.entrenamiento import parser_base, entrenar

RESULTADOS = Path(__file__).resolve().parents[1]/'results'


def argumentos_ppo(p):
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--epocas-ppo', type=int, default=4)
    p.add_argument('--minibatch', type=int, default=64)
    p.add_argument('--target-kl', type=float, default=0.03)
    return p


def actualizar_ppo(red, opt, lote, args):
    metricas = []
    for _ in range(args.epocas_ppo):
        for ix in torch.randperm(len(lote['s']), device=lote['s'].device).split(args.minibatch):
            logp, valor, entropia = red.evaluar(lote['s'][ix], lote['z'][ix])
            logratio = logp-lote['logp'][ix]
            ratio = logratio.exp()
            kl = (ratio-1-logratio).mean()
            if metricas and kl.item() > args.target_kl:
                return np.mean(metricas, axis=0).tolist()
            adv = lote['adv'][ix]
            actor_loss = -torch.minimum(ratio*adv, ratio.clamp(1-args.clip, 1+args.clip)*adv).mean()
            critic_loss = (valor-lote['ret'][ix]).square().mean()
            loss = actor_loss+args.coef_critico*critic_loss-args.entropia*entropia.mean()
            opt.zero_grad()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(red.parameters(), 0.5)
            opt.step()
            metricas.append([actor_loss.item(), critic_loss.item(), entropia.mean().item(), kl.item(),
                             ((ratio-1).abs() > args.clip).float().mean().item(), float(grad)])
    return np.mean(metricas, axis=0).tolist()


if __name__ == '__main__':
    args = argumentos_ppo(parser_base('PPO continuo para TurtleBot3')).parse_args()
    entrenar(args, 'ppo', actualizar_ppo, RESULTADOS)
