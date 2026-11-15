"""MDP continuo sobre ROS 2 Humble / Gazebo Classic (TurtleBot3 burger).

Estado (52): 24 haces LiDAR (minimo por sector / 3.5), 24 velocidades de cierre por sector (/1.5),
             (v, w) reales normalizadas, distancia a la meta y angulo a la meta / pi.
Accion (2):  v en [0, 0.22] m/s y w en [-2.84, 2.84] rad/s, publicada como Twist en /cmd_vel.
Recompensa:  1/max(d_t, 0.20)^2 +100 (meta) o -100 (choque); d_t es la distancia actual a la meta.
Fin:         choque si LiDAR < 0.15 m, exito si d < 0.20 m, truncado a `limite` pasos de dt = 0.2 s.
"""
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

V_MAX, W_MAX = 0.22, 2.84
ESCENARIOS = {
    # Meta cercana y obstaculos lentos fuera del camino.
    'facil': dict(meta=(0.5, 0.0), fases_aleatorias=False,
                  obstaculos=dict(velocidad=0.08, omega=0.5, posiciones=[(0.25, -0.40), (0.85, 0.45)])),
    # Meta lejana y dos cilindros que cruzan la linea robot-meta con fase aleatoria por episodio.
    'cruce': dict(meta=(1.5, 0.0), fases_aleatorias=True,
                  obstaculos=dict(velocidad=0.20, omega=0.4, posiciones=[(0.65, 0.0), (1.10, 0.25)])),
}
TOPICO_FASES = '/lab4_fases'
# Perturbacion inicial (escala 1.0): giro en sitio de hasta +-0.9 rad y avance de hasta 0.8 s.
PERTURBACION_YAW, PERTURBACION_AVANCE = 0.9, 0.8
ACCION_MIN, ACCION_MAX = np.array([0., -W_MAX], np.float32), np.array([V_MAX, W_MAX], np.float32)


class Entorno(Node):
    def __init__(self, escenario='cruce', dt=0.2, limite=150, ruido_lidar=0.0, sincronizar=False,
                 perturbacion=0.0):
        """sincronizar=True pausa Gazebo entre acciones (entrenamiento); False corre en tiempo real (teleop)."""
        super().__init__('lab4_entorno', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.meta = np.asarray(ESCENARIOS[escenario]['meta'], np.float32)
        self.fases_aleatorias = ESCENARIOS[escenario]['fases_aleatorias']
        self.dt, self.limite, self.ruido_lidar = dt, limite, ruido_lidar
        self.sincronizar, self.perturbacion = sincronizar, perturbacion
        self.rng = np.random.default_rng()
        self.scan = self.pose = None
        self.scan_dqn = None
        self.velocidad_real = np.zeros(2, np.float32)
        self.cierre = np.zeros(24, np.float32)
        self.anterior_movimiento = None
        self.scan_stamp = self.odom_stamp = -1
        self.scan_time = self.odom_time = 0.
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_fases = self.create_publisher(Float32MultiArray, TOPICO_FASES, 10)
        self.create_subscription(LaserScan, '/scan', self._scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self._odom, qos_profile_sensor_data)
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        self.pause_client = self.create_client(Empty, '/pause_physics')
        self.unpause_client = self.create_client(Empty, '/unpause_physics')

    # ---------------------------------------------------------------- sensores
    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _scan(self, msg):
        a = np.asarray(msg.ranges, dtype=np.float32)
        if len(a) < 24:
            return
        # Mismo orden y preprocesado frontal que dqn_environment.py del tutorial.
        angulos = msg.angle_min+np.arange(len(a))*msg.angle_increment
        frente = ((angulos >= 0) & (angulos <= math.pi/2)) | ((angulos >= 3*math.pi/2) & (angulos <= 2*math.pi))
        originales = np.where(np.isposinf(a), 3.5, np.where(np.isnan(a), 0., a))
        self.scan_dqn = originales[frente].copy()
        maximo = min(float(msg.range_max), 3.5)
        a = np.clip(np.nan_to_num(a, nan=0., posinf=maximo, neginf=0.), 0., maximo)
        self.scan = np.asarray([x.min() for x in np.array_split(a, 24)], np.float32)
        self.scan_stamp = self._stamp(msg)
        self.scan_time = time.monotonic()

    def _odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.pose = np.array([p.x, p.y, yaw], np.float32)
        self.velocidad_real = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z], np.float32)
        self.odom_stamp = self._stamp(msg)
        self.odom_time = time.monotonic()

    def esperar_sensores(self, timeout=30., despues=0., stamp_min=-1):
        fin = time.monotonic()+timeout
        while rclpy.ok() and time.monotonic() < fin:
            if (self.scan is not None and self.pose is not None
                    and min(self.scan_time, self.odom_time) > despues
                    and min(self.scan_stamp, self.odom_stamp) >= stamp_min):
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise TimeoutError('No llegan sensores nuevos: revisa robot, /clock, /scan, /odom y pausa de Gazebo.')

    def actualizar_movimiento(self):
        """Velocidad de cierre por sector: diferencia de LiDAR entre lecturas, compensando el giro propio."""
        ahora = (self.scan.copy(), self.scan_stamp, float(self.pose[2]))
        if self.anterior_movimiento is not None:
            anterior, stamp, yaw = self.anterior_movimiento
            dt = (self.scan_stamp-stamp)/1e9
            if 0 < dt <= 1.0:
                dyaw = math.atan2(math.sin(ahora[2]-yaw), math.cos(ahora[2]-yaw))
                alineado = np.interp(np.arange(24)+dyaw*24/(2*math.pi), np.arange(24), anterior, period=24)
                estimacion = np.clip((alineado-self.scan)/dt, -1.5, 1.5).astype(np.float32)
            else:
                estimacion = np.zeros(24, np.float32)
            self.cierre = 0.5*self.cierre+0.5*estimacion
        else:
            self.cierre[:] = 0.
        self.anterior_movimiento = ahora

    def observar(self):
        dx, dy = self.meta-self.pose[:2]
        angulo = math.atan2(dy, dx)-float(self.pose[2])
        angulo = math.atan2(math.sin(angulo), math.cos(angulo))
        lidar = np.clip(self.scan+np.random.normal(0., self.ruido_lidar, self.scan.shape), 0., 3.5)
        return np.concatenate((lidar/3.5, self.cierre/1.5,
                               np.clip(self.velocidad_real/np.array([V_MAX, W_MAX]), -1, 1),
                               [math.hypot(dx, dy), angulo/math.pi])).astype(np.float32)

    def observar_dqn(self):
        """Estado original del experto; no sustituye el estado de 52 valores de BC/DAgger."""
        if self.scan_dqn is None or len(self.scan_dqn) != 24:
            cantidad = 0 if self.scan_dqn is None else len(self.scan_dqn)
            raise ValueError(f'El DQN requiere 24 rayos frontales y llegan {cantidad}. '
                             'Usa el LiDAR de 48 muestras del TurtleBot3 del tutorial; no se remuestrea silenciosamente.')
        dx, dy = self.meta-self.pose[:2]
        angulo = math.atan2(dy, dx)-float(self.pose[2])
        angulo = math.atan2(math.sin(angulo), math.cos(angulo))
        return np.concatenate(([math.hypot(dx, dy), angulo], self.scan_dqn)).astype(np.float32)

    # ---------------------------------------------------------------- control de Gazebo
    def parar(self):
        self.pub.publish(Twist())

    def fisica(self, pausada):
        if not self.sincronizar:
            return
        client = self.pause_client if pausada else self.unpause_client
        if not client.wait_for_service(timeout_sec=5.):
            raise TimeoutError('Falta servicio de pausa de Gazebo.')
        future = client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.)
        if not future.done() or future.exception() is not None:
            raise RuntimeError('No se pudo cambiar la pausa de Gazebo.')

    def liberar(self):
        """Deja el robot detenido y la fisica corriendo al terminar."""
        self.parar()
        self.fisica(False)

    def _avanzar(self, hasta, timeout=15.):
        limite = time.monotonic()+timeout
        while self.get_clock().now().nanoseconds < hasta:
            if not rclpy.ok() or time.monotonic() >= limite:
                self.parar()
                raise TimeoutError('El reloj de Gazebo no avanza; comprueba que no este en pausa.')
            rclpy.spin_once(self, timeout_sec=0.02)

    def _mover(self, v, w, duracion):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.pub.publish(msg)
        self._avanzar(self.get_clock().now().nanoseconds+int(duracion*1e9))

    # ---------------------------------------------------------------- interfaz tipo Gym
    def reset(self, semilla=None):
        """Reinicia Gazebo. Con semilla, la fase de obstaculos y la perturbacion son reproducibles."""
        rng = self.rng if semilla is None else np.random.default_rng(semilla)
        self.parar()
        self.fisica(False)
        if not self.reset_client.wait_for_service(timeout_sec=15.):
            raise TimeoutError('Falta /reset_simulation.')
        self.scan = self.pose = None
        self.velocidad_real = np.zeros(2, np.float32)
        self.cierre = np.zeros(24, np.float32)
        self.anterior_movimiento = None
        future = self.reset_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.)
        if not future.done() or future.exception() is not None:
            raise RuntimeError('No se pudo reiniciar Gazebo.')
        fases = None
        if self.fases_aleatorias:  # base/obstaculos.py escucha este topico
            fases = Float32MultiArray(data=[float(x) for x in rng.uniform(0., 2*math.pi, 2)])
            self.pub_fases.publish(fases)
        marca = time.monotonic()
        self.esperar_sensores(despues=marca)
        fin = time.monotonic()+15.
        # Descarta lecturas en cola de antes del reset.
        while (np.linalg.norm(self.pose[:2]) > 0.08
               or abs(self.get_clock().now().nanoseconds-self.odom_stamp) > 500_000_000):
            if time.monotonic() >= fin:
                raise RuntimeError('El reset no devolvio el robot al origen.')
            rclpy.spin_once(self, timeout_sec=0.05)
        if fases is not None:
            self.pub_fases.publish(fases)
        self._avanzar(self.get_clock().now().nanoseconds+300_000_000)
        self.esperar_sensores(despues=marca)
        if self.perturbacion > 0:
            # Perturbacion inicial fuera del episodio: giro en sitio y avance corto.
            yaw = rng.uniform(-1., 1.)*PERTURBACION_YAW*self.perturbacion
            avance = rng.uniform(0., 1.)*PERTURBACION_AVANCE*self.perturbacion
            self._mover(0., math.copysign(1.5, yaw), abs(yaw)/1.5)
            self._mover(0.15, 0., avance)
            self._mover(0., 0., 0.3)
            self.esperar_sensores(timeout=10., stamp_min=self.get_clock().now().nanoseconds)
        self.fisica(True)
        self.actualizar_movimiento()
        self.pasos = 0
        return self.observar()

    def step(self, accion):
        v, w = np.clip(np.asarray(accion, np.float32), ACCION_MIN, ACCION_MAX)
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        inicio = self.get_clock().now().nanoseconds
        self.pub.publish(msg)
        try:
            self.fisica(False)
            objetivo = inicio+int(self.dt*1e9)
            self._avanzar(objetivo)
            self.esperar_sensores(timeout=10., stamp_min=objetivo)
            self.fisica(True)
        except BaseException:
            self.parar()
            raise
        self.actualizar_movimiento()
        obs = self.observar()
        self.pasos += 1
        distancia = float(obs[-2])
        colision = bool(self.scan.min() < 0.15)
        exito = bool(distancia < 0.20 and not colision)
        terminada = colision or exito
        truncada = self.pasos >= self.limite and not terminada
        # El episodio termina dentro de 0.20 m; usar ese radio como piso evita la singularidad en la meta.
        recompensa = 1.0 / max(distancia, 0.20)**2
        if colision:
            recompensa -= 10.
        elif exito:
            recompensa += 10.
        if terminada or truncada:
            self.parar()
        info = dict(exito=exito, colision=colision, x=float(self.pose[0]), y=float(self.pose[1]),
                    v=float(v), omega=float(w), tiempo_sim=self.get_clock().now().nanoseconds/1e9,
                    distancia_meta=distancia, distancia_min=float(self.scan.min()))
        return obs, recompensa, terminada, truncada, info
