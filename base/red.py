"""La unica red de politica del laboratorio: actor gaussiano + critico V(s).

Se usa en todas las partes para que los pesos pasen de una a otra:
  Parte A  A2C y PPO entrenan actor y critico.
  Parte B  BC y DAgger ajustan solo el actor (regresion de la media); el critico no se usa.
  Parte C  RLHF arranca de los pesos de DAgger y la referencia de la penalizacion KL es esa misma red.

Actor:   52 -> 128 -> 128 -> 2 (Tanh), media mu(s) de z; log_std es un parametro libre.
Accion:  z ~ N(mu, sigma);  v = 0.11*(1 + tanh z0),  w = 2.84*tanh z1  (siempre dentro de los limites).
Critico: 52 -> 128 -> 128 -> 1.
"""
import math
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

ESTADO = 52


class ActorCritico(nn.Module):
    def __init__(self, estado=ESTADO):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(estado, 128), nn.Tanh(),
                                   nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
        self.log_std = nn.Parameter(torch.full((2,), -0.7))
        self.critic = nn.Sequential(nn.Linear(estado, 128), nn.Tanh(),
                                    nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        for modulo in self.modules():
            if isinstance(modulo, nn.Linear):
                nn.init.orthogonal_(modulo.weight, math.sqrt(2))
                nn.init.zeros_(modulo.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.)

    def dist(self, s):
        return Normal(self.actor(s), self.log_std.clamp(-4, 1).exp())

    @staticmethod
    def _logp(d, z):
        """log pi(a|s) de la accion fisica: log N(z) menos el Jacobiano de tanh y de la escala."""
        log_jac = 2*(math.log(2)-z-F.softplus(-2*z))
        return (d.log_prob(z)-log_jac).sum(-1)-math.log(0.11*2.84)

    @staticmethod
    def a_fisica(z):
        u = z.tanh()
        return torch.stack(((u[..., 0]+1)*0.11, u[..., 1]*2.84), -1)

    def action(self, s, deterministic=False, ruido=1.0):
        """Devuelve accion fisica (v, w), z, log pi(a|s) y V(s). `ruido` escala la desviacion."""
        d = self.dist(s)
        z = d.mean if deterministic else d.mean+ruido*d.stddev*torch.randn_like(d.mean)
        return self.a_fisica(z), z, self._logp(d, z), self.critic(s).squeeze(-1)

    def evaluar(self, s, z):
        """log pi(z|s), V(s) y estimador Monte Carlo de la entropia (para A2C/PPO)."""
        d = self.dist(s)
        return self._logp(d, z), self.critic(s).squeeze(-1), -self._logp(d, d.rsample())


def dispositivo(nombre='auto'):
    return ('cuda' if torch.cuda.is_available() else 'cpu') if nombre == 'auto' else nombre


def guardar_politica(path, red, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporal = path.with_suffix(path.suffix+'.tmp')
    torch.save(dict(modelo=red.state_dict(), **extra), temporal)
    temporal.replace(path)


def cargar_politica(path, device='cpu'):
    red = ActorCritico().to(device)
    red.load_state_dict(torch.load(path, map_location=device, weights_only=False)['modelo'])
    return red
