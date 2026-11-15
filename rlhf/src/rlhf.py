"""Parte C - Ajuste final con PPO sobre la recompensa aprendida y penalizacion KL hacia pi_DAgger.

    R_RLHF(s, a) = r_psi(s, a) - beta * ln( pi_theta(a|s) / pi_DAgger(a|s) )

    python3 rlhf/src/rlhf.py --corrida beta     --beta 5   # ancla activa
    python3 rlhf/src/rlhf.py --corrida sin_beta --beta 0   # sin ancla: Reward Hacking
pi_theta arranca en pi_DAgger; la actualizacion es la de PPO de la Parte A.
Ademas de los logs de base/entrenamiento.py escribe logs/<corrida>/kl_referencia.csv.
"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ/'ppo'/'src'))
import torch
from base.entrenamiento import parser_base, entrenar
from base.evaluar import csv_append
from base.red import cargar_politica, dispositivo
from ppo import actualizar_ppo, argumentos_ppo
from preferencias import DAGGER, PESOS, RESULTADOS, cargar_recompensa

if __name__ == '__main__':
    p = argumentos_ppo(parser_base('RLHF: PPO sobre r_psi con ancla KL a pi_DAgger'))
    p.add_argument('--beta', type=float, required=True)
    p.add_argument('--referencia', default=str(DAGGER))
    p.add_argument('--recompensa', default=str(PESOS/'recompensa_bt.pth'))
    p.set_defaults(pasos=4000, validar_cada=1000, pruebas_validacion=5, perturbacion=1.0, entropia=0.0)
    args = p.parse_args()
    device = dispositivo()
    red = cargar_politica(args.referencia, device)
    referencia = cargar_politica(args.referencia, device).eval().requires_grad_(False)
    r_psi = cargar_recompensa(args.recompensa, device)
    registro = []  # (r_psi, log-ratio) por paso; cada 64 pasos se escribe la media
    destino = RESULTADOS/'logs'/args.corrida/'kl_referencia.csv'

    def recompensa_rlhf(s, a, z, logp, _r_entorno):
        with torch.no_grad():
            r = float(r_psi.normalizada(s, a))
            log_ratio = float(logp-referencia.evaluar(s, z)[0])
        registro.append((r, log_ratio))
        if len(registro) % 64 == 0:
            bloque = registro[-64:]
            csv_append(destino, ['pasos', 'r_psi_media', 'kl_estimado'],
                       [[len(registro), sum(x[0] for x in bloque)/64, sum(x[1] for x in bloque)/64]])
        return r-args.beta*log_ratio

    entrenar(args, 'rlhf', actualizar_ppo, RESULTADOS, red=red, recompensa_extra=recompensa_rlhf)
