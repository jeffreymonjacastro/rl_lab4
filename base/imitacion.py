"""Parte B - piezas comunes de BC y DAgger: teclado WASD, sesion interactiva, datasets y ajuste supervisado.

La red es base/red.py (ActorCritico). BC y DAgger solo entrenan el actor: regresion MSE de tanh(mu(s))
contra la accion del experto normalizada a [-1, 1]. Se usa la misma red que en A/C para que la Parte C
pueda arrancar RLHF desde dagger_sim.pth.

Teclado (ventana Tk "Lab 4", debe tener el foco):
  W avanza | A/D giran en el sitio | W+A / W+D curva | S frena | flechas equivalen | R repite | Q termina
"""
import sys
import time
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import csv
import tkinter
import numpy as np
import torch

ESCENARIO = 'cruce'
SEMILLA_EVAL = 7000   # mismos episodios perturbados para BC y cada ronda de DAgger
TRAY_HEADER = ['episodio', 'paso', 'x', 'y', 'humano', 'v', 'omega', 'exito', 'colision']

# ------------------------------------------------------------------ teclado -> (v, w)
ALIAS = {'up': 'w', 'left': 'a', 'down': 's', 'right': 'd', 'space': ' '}
MOVIMIENTO = {'w', 'a', 's', 'd'}
V_TELEOP, W_GIRO, W_CURVA = 0.18, 1.2, 0.8
SUAVIZADO = 0.5  # filtro de primer orden: evita escalones bruscos en las etiquetas


def accion_teclas(teclas, previa):
    v = V_TELEOP if 'w' in teclas and 's' not in teclas else 0.0
    giro = ('a' in teclas)-('d' in teclas)
    objetivo = np.array([v, giro*(W_CURVA if v > 0 else W_GIRO)], np.float32)
    return (SUAVIZADO*np.asarray(previa, np.float32)+(1-SUAVIZADO)*objetivo).astype(np.float32)


class Teclado:
    """Teclas mantenidas (con combinaciones) mediante una ventana Tk; filtra la autorepeticion de X11."""

    def __init__(self, titulo):
        self.tk = tkinter.Tk()
        self.tk.title(titulo)
        self.tk.geometry('560x230+40+40')
        self.tk.attributes('-topmost', True)
        self.modo = tkinter.Label(self.tk, font=('DejaVu Sans', 22, 'bold'), fg='white', bg='#444444')
        self.modo.pack(fill='x')
        self.texto = tkinter.Label(self.tk, font=('DejaVu Sans Mono', 11), justify='left', anchor='w')
        self.texto.pack(fill='both', expand=True, padx=10, pady=6)
        tkinter.Label(self.tk, text='W/A/S/D o flechas: conducir | R: repetir episodio | Q: terminar',
                      font=('DejaVu Sans', 9), fg='#555555').pack(side='bottom')
        self.abajo, self.eventos, self._pendientes, self.cerrado = set(), [], {}, False
        self.tk.bind_all('<KeyPress>', self._pulsar)
        self.tk.bind_all('<KeyRelease>', self._soltar)
        self.tk.protocol('WM_DELETE_WINDOW', lambda: setattr(self, 'cerrado', True))
        self.tk.focus_force()
        self.tk.update()

    def _pulsar(self, evento):
        k = ALIAS.get(evento.keysym.lower(), evento.keysym.lower())
        if k in self._pendientes:  # autorepeticion: soltar+pulsar inmediato
            self.tk.after_cancel(self._pendientes.pop(k))
        if k not in self.abajo:
            self.abajo.add(k)
            self.eventos.append(k)

    def _soltar(self, evento):
        k = ALIAS.get(evento.keysym.lower(), evento.keysym.lower())
        if k in self._pendientes:
            self.tk.after_cancel(self._pendientes[k])
        self._pendientes[k] = self.tk.after(60, lambda: (self.abajo.discard(k), self._pendientes.pop(k, None)))

    def leer(self):
        """(teclas mantenidas o pulsadas desde la ultima lectura, eventos nuevos)."""
        if self.cerrado:
            return set(), ['q']
        self.tk.update()
        eventos, self.eventos = self.eventos, []
        return set(self.abajo) | set(eventos), eventos

    def estado(self, modo, color, texto):
        if not self.cerrado:
            self.modo.config(text=modo, bg=color)
            self.texto.config(text=texto)
            self.tk.update()

    def esperar(self, tecla, modo, texto):
        """Bloquea hasta `tecla` (True) o Q / cierre (False)."""
        self.estado(modo, '#6250d6', texto)
        self.eventos.clear()
        while True:
            _, eventos = self.leer()
            if tecla in eventos:
                return True
            if 'q' in eventos or 'escape' in eventos:
                return False
            time.sleep(0.03)

    def cerrar(self):
        if not self.cerrado:
            self.cerrado = True
            self.tk.destroy()


# ------------------------------------------------------------------ datos
def cargar_datos(path):
    with np.load(path) as d:
        return d['estados'].astype(np.float32), d['acciones'].astype(np.float32)


def guardar_datos(path, estados, acciones):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, estados=np.asarray(estados, np.float32).reshape(-1, 52),
                        acciones=np.asarray(acciones, np.float32).reshape(-1, 2))


def escribir_csv(path, header, filas, agregar=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a' if agregar else 'w', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(header)
        w.writerows(filas)


# ------------------------------------------------------------------ aprendizaje supervisado
def ajustar(red, estados, acciones, epocas, lr, device):
    """MSE entre tanh(mu(s)) y la accion experta normalizada; Adam, minibatch 128."""
    s = torch.as_tensor(estados, dtype=torch.float32, device=device)
    a = torch.as_tensor(acciones, dtype=torch.float32, device=device)
    objetivo = torch.stack((a[:, 0]/0.11-1, a[:, 1]/2.84), 1).clamp(-1, 1)
    opt = torch.optim.Adam(red.actor.parameters(), lr=lr)
    historial = []
    red.train()
    for epoca in range(1, epocas+1):
        total = 0.
        for ix in torch.randperm(len(s), device=device).split(128):
            loss = (red.actor(s[ix]).tanh()-objetivo[ix]).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()*len(ix)
        historial.append((epoca, total/len(s)))
    with torch.no_grad():
        red.log_std.fill_(-1.0)  # desviacion fija: la usa la Parte C (muestreo y KL)
    print(f'Ajuste: {len(s)} pares, {epocas} epocas, MSE {historial[0][1]:.4f} -> {historial[-1][1]:.4f}', flush=True)
    return historial


# ------------------------------------------------------------------ sesion interactiva
def sesion(env, teclado, red, episodios, device, titulo, semilla_base=None, solo_exitos=False):
    """red=None: teleoperacion pura (demostraciones D0).
    red dada: la politica pilotea; mantener W/A/S/D toma el control y al soltar vuelve la politica
    (compuerta HG-DAgger). En ambos casos solo se guardan los pasos con tecla pulsada.
    Devuelve (estados, acciones, filas de trayectoria, estadisticas, abortado)."""
    estados, acciones, trayectoria = [], [], []
    st = dict(pasos=0, pasos_humanos=0, intervenciones=0, episodios=0, colisiones=0, exitos=0)
    ep = 0
    while ep < episodios:
        obs = env.reset(None if semilla_base is None else semilla_base+ep+1)
        e_s, e_a, e_t = [], [], []
        previa = np.zeros(2, np.float32)
        humano_antes, n_humano, n_interv, paso, repetir = False, 0, 0, 0, False
        teclado.eventos.clear()
        while True:
            teclas, eventos = teclado.leer()
            if 'q' in eventos or 'escape' in eventos:
                return estados, acciones, trayectoria, st, True
            if 'r' in eventos:
                repetir = True
                break
            mov = teclas & MOVIMIENTO
            if mov or red is None:
                a = accion_teclas(mov, previa)
            else:
                with torch.no_grad():
                    a = red.action(torch.as_tensor(obs, device=device), True)[0].cpu().numpy()
            if mov:
                e_s.append(obs.copy())
                e_a.append(a.copy())
                n_humano += 1
                n_interv += int(not humano_antes)
            humano_antes = bool(mov)
            obs, _, done, trunc, info = env.step(a)
            previa = a
            paso += 1
            e_t.append([ep+1, paso, info['x'], info['y'], int(bool(mov)), info['v'], info['omega']])
            teclado.estado('HUMANO' if mov else ('ESPERANDO TECLA' if red is None else 'POLITICA'),
                           '#d03b3b' if mov else ('#8a8980' if red is None else '#0ca30c'),
                           f'{titulo}\nEpisodio {ep+1}/{episodios}   paso {paso}\n'
                           f'Distancia a meta {info["distancia_meta"]:.2f} m   obstaculo mas cercano '
                           f'{info["distancia_min"]:.2f} m\nIntervenciones {st["intervenciones"]+n_interv}   '
                           f'pasos humanos {st["pasos_humanos"]+n_humano}   colisiones {st["colisiones"]}')
            if done or trunc:
                break
        if repetir or (solo_exitos and not info['exito']):
            print('Episodio descartado' + ('' if repetir else ' (no llego a la meta)'), flush=True)
            continue
        ep += 1
        estados += e_s
        acciones += e_a
        trayectoria += [f+[info['exito'], info['colision']] for f in e_t]
        for clave, valor in [('pasos', paso), ('pasos_humanos', n_humano), ('intervenciones', n_interv),
                             ('episodios', 1), ('colisiones', int(info['colision'])), ('exitos', int(info['exito']))]:
            st[clave] += valor
        print(f'Episodio {ep}/{episodios}: pasos={paso}, humano={n_humano}, exito={info["exito"]}, '
              f'colision={info["colision"]}', flush=True)
    return estados, acciones, trayectoria, st, False
