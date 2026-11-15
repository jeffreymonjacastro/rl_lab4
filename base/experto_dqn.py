"""Inferencia PyTorch de las capas Dense del DQN externo, sin lanzar otro controlador ROS.

La arquitectura y los pesos se leen del HDF5 original de Keras; no se entrena ni modifica el experto.
Entrada original: distancia, angulo (radianes), 24 rayos frontales en metros.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODELO_DQN = (Path(__file__).resolve().parents[2]/'turtlebot3_machine_learning'/
              'turtlebot3_dqn'/'saved_model'/'model1.h5')
GIROS = np.array([1.5, 0.75, 0., -0.75, -1.5], dtype=np.float32)


class ExpertoDQN:
    def __init__(self, archivo=MODELO_DQN):
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError('Falta h5py para leer el experto: python3 -m pip install "h5py>=3"') from e
        self.archivo = Path(archivo).resolve()
        capas = []
        anterior = 26
        with h5py.File(self.archivo, 'r') as f:
            config = json.loads(f.attrs['model_config'])
            for capa in config['config']['layers']:
                tipo, c = capa['class_name'], capa['config']
                if tipo == 'InputLayer':
                    forma = c.get('batch_shape', c.get('batch_input_shape'))
                    if forma[-1] != 26:
                        raise ValueError(f'El DQN debe recibir 26 valores, no {forma}.')
                    continue
                if tipo != 'Dense' or c.get('activation') not in ('relu', 'linear'):
                    raise ValueError(f'Capa del experto no soportada: {tipo} / {c.get("activation")}')
                grupo = f['model_weights'][c['name']]
                encontrados = {}
                def leer(nombre, obj):
                    if isinstance(obj, h5py.Dataset):
                        encontrados[nombre.rsplit('/', 1)[-1].split(':')[0]] = np.asarray(obj, np.float32)
                grupo.visititems(leer)
                kernel, bias = encontrados['kernel'], encontrados['bias']
                if kernel.shape != (anterior, c['units']) or bias.shape != (c['units'],):
                    raise ValueError(f'Dimensiones incompatibles en {c["name"]}: {kernel.shape}, {bias.shape}')
                lineal = nn.Linear(anterior, c['units'])
                with torch.no_grad():
                    lineal.weight.copy_(torch.from_numpy(kernel.T.copy()))
                    lineal.bias.copy_(torch.from_numpy(bias))
                capas.append(lineal)
                if c['activation'] == 'relu':
                    capas.append(nn.ReLU())
                anterior = c['units']
        if anterior != len(GIROS):
            raise ValueError('El experto debe producir cinco valores Q.')
        self.red = nn.Sequential(*capas).eval().requires_grad_(False)

    def valores_q(self, estado):
        estado = np.asarray(estado, np.float32)
        if estado.shape != (26,) or not np.isfinite(estado).all():
            raise ValueError(f'Estado DQN invalido: se esperan 26 valores finitos, recibidos {estado.shape}.')
        with torch.no_grad():
            return self.red(torch.from_numpy(estado)).numpy()

    def accion(self, estado):
        indice = int(np.argmax(self.valores_q(estado)))
        return np.array([0.2, GIROS[indice]], dtype=np.float32)
