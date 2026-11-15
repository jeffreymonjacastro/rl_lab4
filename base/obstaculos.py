"""Dos cilindros moviles y el marcador verde de la meta en Gazebo.

    python3 base/obstaculos.py --escenario cruce      # Ctrl+C (o SIGTERM) los retira
Cada cilindro oscila en y: v = velocidad*cos(omega*t + fase). En 'cruce' el entorno publica una fase
nueva por episodio en /lab4_fases, asi cada reset genera un cruce distinto (reproducible con semilla).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import math
import signal
import time
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from std_msgs.msg import Float32MultiArray
from base.entorno import ESCENARIOS, TOPICO_FASES

CILINDRO = """<sdf version="1.6"><model name="{nombre}"><link name="body">
<inertial><mass>2</mass><inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.02</izz></inertia></inertial>
<collision name="colision"><geometry><cylinder><radius>0.12</radius><length>0.35</length></cylinder></geometry></collision>
<visual name="visual"><geometry><cylinder><radius>0.12</radius><length>0.35</length></cylinder></geometry>
<material><ambient>0.9 0.12 0.05 1</ambient><diffuse>0.9 0.12 0.05 1</diffuse></material></visual></link>
<plugin name="movimiento" filename="libgazebo_ros_planar_move.so"><ros><namespace>/{nombre}</namespace></ros>
<update_rate>50</update_rate><publish_rate>10</publish_rate><publish_odom>true</publish_odom>
<publish_odom_tf>false</publish_odom_tf><odometry_frame>world</odometry_frame><robot_base_frame>body</robot_base_frame>
</plugin></model></sdf>"""
VERDE = '<material><ambient>0.05 1 0.05 1</ambient><diffuse>0.05 1 0.05 1</diffuse><emissive>0.05 0.5 0.05 1</emissive></material>'
META = f"""<sdf version="1.6"><model name="lab4_meta"><static>true</static><link name="marcador">
<visual name="aro"><pose>0 0 0.015 0 0 0</pose><geometry><cylinder><radius>0.2</radius><length>0.02</length></cylinder></geometry>{VERDE}</visual>
<visual name="poste"><pose>0 0 0.4 0 0 0</pose><geometry><cylinder><radius>0.025</radius><length>0.8</length></cylinder></geometry>{VERDE}</visual>
<visual name="bandera"><pose>0 0 0.83 0 0 0</pose><geometry><sphere><radius>0.09</radius></sphere></geometry>{VERDE}</visual>
</link></model></sdf>"""
NOMBRES = ['lab4_obstaculo_1']


class Obstaculos(Node):
    def __init__(self, escenario):
        super().__init__('lab4_obstaculos', parameter_overrides=[Parameter('use_sim_time', value=True)])
        perfil = ESCENARIOS[escenario]['obstaculos']
        self.meta = ESCENARIOS[escenario]['meta']
        self.velocidad, self.omega, self.posiciones = perfil['velocidad'], perfil['omega'], perfil['posiciones']
        self.fases = [0.0, math.pi]
        self.create_subscription(Float32MultiArray, TOPICO_FASES, self._fases, 10)
        self.spawn = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete = self.create_client(DeleteEntity, '/delete_entity')
        self.pubs = [self.create_publisher(Twist, f'/{n}/cmd_vel', 10) for n in NOMBRES]
        self.timer = None
        self.creados = []

    def _fases(self, msg):
        if len(msg.data) == len(self.fases):
            self.fases = [float(x) for x in msg.data]

    def _retirar(self, nombre, timeout=10):
        req = DeleteEntity.Request()
        req.name = nombre
        f = self.delete.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=timeout)
        return f.done()

    def _insertar(self, nombre, xml, x, y, z=0.):
        self._retirar(nombre)  # por si quedo de una ejecucion anterior
        req = SpawnEntity.Request()
        req.name, req.xml = nombre, xml
        req.initial_pose.position.x, req.initial_pose.position.y, req.initial_pose.position.z = float(x), float(y), z
        req.initial_pose.orientation.w = 1.0
        f = self.spawn.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=20)
        if not f.done() or not f.result() or not f.result().success:
            raise RuntimeError(f'No se pudo crear {nombre}')
        self.creados.append(nombre)

    def _esperar_servicios(self, timeout):
        clientes = {'/spawn_entity': self.spawn, '/delete_entity': self.delete}
        hasta = time.monotonic()+timeout
        siguiente_aviso = 0.
        while rclpy.ok():
            pendientes = [nombre for nombre, cliente in clientes.items() if not cliente.service_is_ready()]
            if not pendientes:
                return
            ahora = time.monotonic()
            if ahora >= hasta:
                visibles = dict(self.get_service_names_and_types())
                detalle = ', '.join(f'{nombre} ({"anunciado, cliente sin conectar" if nombre in visibles else "no descubierto"})'
                                    for nombre in pendientes)
                raise RuntimeError(f'No se pudo conectar en {timeout:g} s: {detalle}. '
                                   'Carga base/ros.sh 2 tambien en la terminal de Gazebo antes de iniciarlo.')
            if ahora >= siguiente_aviso:
                print('Esperando conexion con Gazebo:', ', '.join(pendientes), flush=True)
                siguiente_aviso = ahora+5.
            # Procesa tambien callbacks y eventos del nodo durante el descubrimiento.
            rclpy.spin_once(self, timeout_sec=min(0.1, hasta-ahora))
        raise RuntimeError('ROS se cerro mientras se esperaban los servicios de Gazebo.')

    def crear(self, timeout=60.):
        self._esperar_servicios(timeout)
        for nombre, (x, y) in zip(NOMBRES, self.posiciones):
            self._insertar(nombre, CILINDRO.format(nombre=nombre), x, y, 0.18)
        self._insertar('lab4_meta', META, *self.meta)
        print('Meta visible en', self.meta, flush=True)
        self.timer = self.create_timer(0.05, self.mover, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def mover(self):
        t = self.get_clock().now().nanoseconds/1e9
        for pub, fase in zip(self.pubs, self.fases):
            msg = Twist()
            msg.linear.y = self.velocidad*math.cos(self.omega*t+fase)
            pub.publish(msg)

    def borrar(self):
        if not self.creados:
            return
        for pub in self.pubs:
            pub.publish(Twist())
        if self.delete.wait_for_service(timeout_sec=3):
            for nombre in self.creados:
                self._retirar(nombre, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--escenario', choices=list(ESCENARIOS), default='cruce')
    p.add_argument('--espera-servicios', type=float, default=60., help='segundos para conectar con Gazebo')
    args = p.parse_args()
    if args.espera_servicios <= 0:
        p.error('--espera-servicios debe ser mayor que cero')
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))  # kill = Ctrl+C
    signal.signal(signal.SIGINT, signal.default_int_handler)
    n = Obstaculos(args.escenario)
    try:
        n.crear(args.espera_servicios)
        print('Obstaculos en movimiento. Ctrl+C para detener y retirar.', flush=True)
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.borrar()
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
