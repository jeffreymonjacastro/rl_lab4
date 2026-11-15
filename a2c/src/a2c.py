"""Parte A - Advantage Actor-Critic (A2C): un unico paso de gradiente por rollout de 64 transiciones.

    L(theta, phi) = -E[log pi(a|s) * A_GAE] + c_v * E[(V(s) - R)^2] - c_h * H[pi]

    python3 a2c/src/a2c.py --corrida cruce --escenario cruce --pasos 20000
Salida: a2c/results/{weights,logs}/<corrida>/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from base.entrenamiento import parser_base, entrenar
from base.red import cargar_politica, dispositivo

RESULTADOS = Path(__file__).resolve().parents[1]/'results'


def actualizar_a2c(red, opt, lote, args):
    logp, valor, entropia = red.evaluar(lote['s'], lote['z'])
    logratio = logp-lote['logp']
    actor_loss = -(logp*lote['adv']).mean()
    critic_loss = (valor-lote['ret']).square().mean()
    loss = actor_loss+args.coef_critico*critic_loss-args.entropia*entropia.mean()
    opt.zero_grad()
    loss.backward()
    grad = nn.utils.clip_grad_norm_(red.parameters(), 0.5)
    opt.step()
    kl = (logratio.exp()-1-logratio).mean()  # solo diagnostico
    return [actor_loss.item(), critic_loss.item(), entropia.mean().item(), kl.item(), 0., float(grad)]


if __name__ == '__main__':
    parser = parser_base('A2C continuo para TurtleBot3')
    parser.add_argument('--inicial', help='pesos de una politica existente; inicia un optimizador nuevo')
    args = parser.parse_args()
    if args.inicial and not Path(args.inicial).is_file():
        parser.error(f'No existe el archivo de pesos: {args.inicial}')
    red = cargar_politica(args.inicial, dispositivo()) if args.inicial else None
    entrenar(args, 'a2c', actualizar_a2c, RESULTADOS, red=red)
