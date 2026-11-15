"""Parte C - Modelo de recompensa por preferencias (Bradley-Terry).

    python3 rlhf/src/preferencias.py recoger --pares 20   # 2 trayectorias por par con pi_DAgger + ruido
    python3 rlhf/src/etiquetar.py                          # el humano elige la mejor de cada par
    python3 rlhf/src/preferencias.py entrenar              # L_RM = -ln sigma(R(tau_w) - R(tau_l))
Las dos trayectorias de un par comparten escenario (misma semilla) y difieren en el ruido de la accion.
r_psi(s, a): MLP 54 -> 128 -> 128 -> 1; R(tau) = suma de r_psi sobre la trayectoria.
"""
import sys
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
import argparse
import csv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RESULTADOS = RAIZ/'rlhf'/'results'
DATOS, PESOS, LOGS = RESULTADOS/'data', RESULTADOS/'weights', RESULTADOS/'logs'
DAGGER = RAIZ/'dagger'/'results'/'weights'/'dagger_sim.pth'
ESCENARIO = 'cruce'
# Escala del ruido de exploracion sobre la desviacion de pi_DAgger. Con valores grandes (2.0) ninguna
# trayectoria llega a la meta y r_psi solo aprende a evitar choques.
RUIDOS = [0.3, 0.6, 1.0]


class ModeloRecompensa(nn.Module):
    """r_psi(s, a): estado (52) y accion normalizada (2) -> recompensa escalar."""

    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(54, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(),
                                 nn.Linear(128, 1))
        self.register_buffer('media', torch.zeros(()))
        self.register_buffer('escala', torch.ones(()))

    def forward(self, estados, acciones):
        a = torch.stack((acciones[..., 0]/0.11-1, acciones[..., 1]/2.84), -1)
        return self.red(torch.cat([estados, a], -1)).squeeze(-1)

    def normalizada(self, estados, acciones):
        """Recompensa con media 0 y desviacion 1 sobre los datos de preferencias (la usa RLHF)."""
        return (self(estados, acciones)-self.media)/self.escala


def cargar_recompensa(path, device):
    modelo = ModeloRecompensa().to(device)
    modelo.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return modelo.eval()


def recoger(args):
    import rclpy
    from base.entorno import Entorno
    from base.red import cargar_politica, dispositivo
    device = dispositivo()
    carpeta = DATOS/'trayectorias'
    carpeta.mkdir(parents=True, exist_ok=True)
    red = cargar_politica(args.modelo, device).eval()
    torch.manual_seed(args.semilla+args.desde)
    rclpy.init()
    env = Entorno(escenario=ESCENARIO, limite=args.pasos, perturbacion=1.0, sincronizar=True)
    try:
        env.esperar_sensores()
        hasta = args.pares if args.hasta is None else min(args.hasta, args.pares)
        for k in range(2*args.desde, 2*hasta):
            path = carpeta/f'trayectoria_{k:03d}.npz'
            if path.exists():
                continue
            # Las dos trayectorias de un par comparten escenario (misma semilla) y difieren en la accion.
            obs = env.reset(args.semilla+k//2)
            ruido = RUIDOS[np.random.randint(len(RUIDOS))]
            datos = dict(estados=[], acciones=[], xy=[], dmin=[], recompensas=[])
            for _ in range(args.pasos):
                with torch.no_grad():
                    a = red.action(torch.as_tensor(obs, device=device), ruido=ruido)[0].cpu().numpy()
                datos['estados'].append(obs)
                datos['acciones'].append(a)
                obs, r, term, trunc, info = env.step(a)
                datos['xy'].append([info['x'], info['y']])
                datos['dmin'].append(info['distancia_min'])
                datos['recompensas'].append(r)
                if term or trunc:
                    break
            np.savez_compressed(path, **{c: np.asarray(v, np.float32) for c, v in datos.items()},
                                colision=np.asarray(info['colision']), exito=np.asarray(info['exito']),
                                meta=env.meta, ruido=np.asarray(ruido))
            print(path.name, 'pasos', len(datos['estados']), 'exito', info['exito'], 'colision',
                  info['colision'], 'ruido', ruido, flush=True)
    finally:
        try:
            env.liberar()
        finally:
            env.destroy_node()
            rclpy.shutdown()


def entrenar(args):
    from base.red import dispositivo
    device = dispositivo()
    with open(args.etiquetas, newline='') as f:
        pares = [(fila['ganadora'], fila['perdedora']) for fila in csv.DictReader(f)]
    if not pares:
        raise SystemExit('No hay pares etiquetados.')

    def tensores(nombre):
        with np.load(DATOS/'trayectorias'/nombre) as d:
            return (torch.as_tensor(d['estados'], device=device), torch.as_tensor(d['acciones'], device=device))
    datos = [(tensores(w), tensores(l)) for w, l in pares]
    torch.manual_seed(args.semilla)
    modelo = ModeloRecompensa().to(device)
    opt = torch.optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    historial = []
    for epoca in range(1, args.epocas+1):
        # R(tau) = suma de r_psi(s, a) sobre la trayectoria.
        dif = torch.stack([modelo(*w).sum()-modelo(*l).sum() for w, l in datos])
        loss = F.softplus(-dif).mean()  # -ln sigma(R_w - R_l)
        opt.zero_grad()
        loss.backward()
        opt.step()
        historial.append((epoca, loss.item(), (dif > 0).float().mean().item()))
        if epoca % 25 == 0 or epoca == 1:
            print(f'Epoca {epoca}: L_RM={loss.item():.4f} acierto={historial[-1][2]:.0%}', flush=True)
    with torch.no_grad():
        r = torch.cat([modelo(*t) for par in datos for t in par])
        modelo.media.copy_(r.mean())
        modelo.escala.copy_(r.std().clamp_min(1e-6))
    PESOS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), PESOS/'recompensa_bt.pth')
    with open(LOGS/'bt_perdida.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoca', 'perdida', 'acierto'])
        w.writerows(historial)
    print(f'{len(pares)} pares. Recompensa guardada en', PESOS/'recompensa_bt.pth')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('modo', choices=['recoger', 'entrenar'])
    p.add_argument('--modelo', default=str(DAGGER))
    p.add_argument('--pares', type=int, default=20)
    p.add_argument('--pasos', type=int, default=80,
                   help='longitud maxima (80 pasos = 16 s; pi_DAgger llega en ~40-60 pasos)')
    p.add_argument('--desde', type=int, default=0, help='primer par (para repartir entre procesos)')
    p.add_argument('--hasta', type=int, default=None, help='par final, exclusivo')
    p.add_argument('--etiquetas', default=str(DATOS/'preferencias.csv'))
    p.add_argument('--epocas', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--semilla', type=int, default=9000)
    args = p.parse_args()
    np.random.seed(args.semilla+args.desde)
    recoger(args) if args.modo == 'recoger' else entrenar(args)
