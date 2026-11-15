"""Pruebas sin iniciar ROS, Gazebo, teleoperacion ni entrenamientos del laboratorio."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from base.entorno import Entorno
from base.experto_dqn import ExpertoDQN, MODELO_DQN, GIROS
from base.imitacion import sesion


class ParteB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experto = ExpertoDQN(MODELO_DQN)

    def test_inferencia_equivale_a_las_capas_keras(self):
        import h5py
        rng = np.random.default_rng(42)
        estado = rng.uniform(0, 3.5, (8, 26)).astype(np.float32)
        esperado = estado.copy()
        with h5py.File(MODELO_DQN, 'r') as f:
            config = json.loads(f.attrs['model_config'])
            for layer in config['config']['layers']:
                if layer['class_name'] != 'Dense':
                    continue
                c = layer['config']
                arrays = {}
                def leer(nombre, obj):
                    if isinstance(obj, h5py.Dataset):
                        arrays[nombre.rsplit('/', 1)[-1].split(':')[0]] = obj[()]
                f['model_weights'][c['name']].visititems(leer)
                esperado = esperado @ arrays['kernel']+arrays['bias']
                if c['activation'] == 'relu':
                    esperado = np.maximum(esperado, 0)
        actual = np.stack([self.experto.valores_q(s) for s in estado])
        np.testing.assert_allclose(actual, esperado, rtol=2e-5, atol=2e-5)

    def test_acciones_originales_y_limites(self):
        s = np.r_[1.5, 0., np.full(24, 3.5)]
        a = self.experto.accion(s)
        np.testing.assert_allclose(a, [0.2, GIROS[np.argmax(self.experto.valores_q(s))]])
        self.assertLessEqual(abs(a[1]), 2.84)
        with self.assertRaises(ValueError):
            self.experto.valores_q(np.zeros(52))

    def test_scan_reproduce_el_estado_frontal_del_tutorial(self):
        raw = np.linspace(.2, 3.5, 48).astype(np.float32)
        raw[0], raw[-1] = np.inf, np.nan
        scan = SimpleNamespace(ranges=raw, angle_min=0., angle_increment=6.28/47,
                               range_max=3.5, header=SimpleNamespace(stamp=SimpleNamespace(sec=2, nanosec=0)))
        fake = SimpleNamespace(_stamp=Entorno._stamp, meta=np.array([1.5, 0.]), pose=np.zeros(3))
        Entorno._scan(fake, scan)
        esperado = []
        for i, valor in enumerate(raw):
            angulo = scan.angle_min+i*scan.angle_increment
            if 0 <= angulo <= np.pi/2 or 3*np.pi/2 <= angulo <= 2*np.pi:
                esperado.append(3.5 if np.isposinf(valor) else 0. if np.isnan(valor) else valor)
        np.testing.assert_allclose(fake.scan_dqn, esperado)
        self.assertEqual(fake.scan.shape, (24,))
        state = Entorno.observar_dqn(fake)
        np.testing.assert_allclose(state[:2], [1.5, 0])
        self.assertEqual(state.shape, (26,))
        fake.scan_dqn = np.zeros(180)
        with self.assertRaises(ValueError):
            Entorno.observar_dqn(fake)

    def test_dagger_solo_guarda_acciones_humanas_con_el_estado_anterior(self):
        class Teclado:
            eventos = []
            teclas = iter([{'w'}, set(), {'s'}])
            def leer(self): return next(self.teclas), []
            def estado(self, *args): pass
        class Env:
            k = 0
            def reset(self, seed): return np.zeros(52, np.float32)
            def step(self, a):
                self.k += 1
                info = dict(x=float(self.k), y=0., v=float(a[0]), omega=float(a[1]),
                            exito=self.k == 3, colision=False, distancia_meta=1., distancia_min=1.)
                return np.full(52, self.k, np.float32), 0., self.k == 3, False, info
        red = SimpleNamespace(action=lambda s, deterministic: (torch.tensor([.2, .75]),))
        ss, aa, tray, stats, abortado = sesion(Env(), Teclado(), red, 1, 'cpu', 'prueba')
        self.assertFalse(abortado)
        self.assertEqual(len(ss), 2)
        self.assertEqual(len(aa), 2)
        np.testing.assert_array_equal(ss[0], np.zeros(52))
        np.testing.assert_array_equal(ss[1], np.full(52, 2))
        self.assertEqual(stats['pasos'], 3)
        self.assertEqual(stats['pasos_humanos'], 2)
        self.assertEqual(stats['intervenciones'], 2)
        self.assertEqual(len(tray), 3)


if __name__ == '__main__':
    unittest.main()
