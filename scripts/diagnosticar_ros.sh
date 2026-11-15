#!/usr/bin/env bash
# Diagnostico de solo lectura: no reinicia Gazebo ni entrena politicas.
cd "$(dirname "$0")/.." || exit 1
source base/ros.sh "${1:-2}"
python3 - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

keys = (
    'ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY', 'RMW_IMPLEMENTATION',
    'ROS_DISCOVERY_SERVER', 'ROS_SUPER_CLIENT',
    'FASTRTPS_DEFAULT_PROFILES_FILE', 'FASTDDS_DEFAULT_PROFILES_FILE',
    'RMW_FASTRTPS_USE_QOS_FROM_XML', 'CYCLONEDDS_URI',
    'GAZEBO_MASTER_URI', 'GAZEBO_IP', 'AMENT_PREFIX_PATH',
    'LD_LIBRARY_PATH', 'PYTHONPATH',
)
probe = r'''
import json, sys, time
import rclpy
from gazebo_msgs.srv import GetModelList, SpawnEntity, DeleteEntity
result = {'python': sys.executable, 'rclpy': rclpy.__file__}
node = None
try:
    rclpy.init()
    result['rmw'] = rclpy.get_rmw_implementation_identifier()
    node = rclpy.create_node('lab4_diagnostico', start_parameter_services=False)
    clients = {name: node.create_client(typ, name) for name, typ in (
        ('/get_model_list', GetModelList), ('/spawn_entity', SpawnEntity),
        ('/delete_entity', DeleteEntity))}
    until = time.monotonic()+12
    while rclpy.ok() and time.monotonic() < until:
        if all(c.service_is_ready() for c in clients.values()):
            break
        rclpy.spin_once(node, timeout_sec=.1)
    result['clientes'] = {n: c.service_is_ready() for n, c in clients.items()}
    result['servicios'] = node.get_service_names_and_types()
    result['topicos'] = node.get_topic_names_and_types()
    result['nodos'] = node.get_node_names_and_namespaces()
    result['servidores_gazebo'] = {
        name: services for name, ns in result['nodos'] if 'gazebo' in name
        for services in [node.get_service_names_and_types_by_node(name, ns)]}
    if clients['/get_model_list'].service_is_ready():
        f = clients['/get_model_list'].call_async(GetModelList.Request())
        rclpy.spin_until_future_complete(node, f, timeout_sec=5)
        if f.done() and f.exception() is None:
            result['modelos'] = list(f.result().model_names)
            result['respuesta_valida'] = bool(f.result().success)
        else:
            result['error_respuesta'] = str(f.exception()) if f.done() else 'sin respuesta en 5 s'
except Exception as e:
    result['error'] = repr(e)
finally:
    if node is not None:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
print('RESULTADO_JSON='+json.dumps(result), flush=True)
'''

sim = os.environ['LAB4_SIM']
out = Path(f'/tmp/rl_pc2_lab4_ros_sim{sim}.json')
report = {'terminal': {k: os.environ.get(k) for k in keys}, 'gazebo': []}
report['namespaces_terminal'] = {k: os.readlink(f'/proc/self/ns/{k}') for k in ('net', 'ipc', 'mnt')}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        if (p/'comm').read_text().strip() != 'gzserver':
            continue
        env = dict(x.split('=', 1) for x in (p/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
        maps = (p/'maps').read_text()
    except (OSError, ProcessLookupError):
        continue
    libs = sorted({line.split()[-1] for line in maps.splitlines()
                   if any(s in line for s in ('librmw_', 'libfastrtps', 'libddsc'))})
    report['gazebo'].append({'pid': p.name, 'entorno': {k: env.get(k) for k in keys}, 'bibliotecas': libs,
                            'namespaces': {k: os.readlink(str(p/'ns'/k)) for k in ('net', 'ipc', 'mnt')}})

def run(label, env):
    print(f'Probando {label} (hasta 22 segundos)...', flush=True)
    try:
        r = subprocess.run([sys.executable, '-u', '-c', probe], env=env,
                           capture_output=True, text=True, timeout=22)
        data = next((json.loads(line.split('=', 1)[1]) for line in r.stdout.splitlines()
                     if line.startswith('RESULTADO_JSON=')), {})
        data.update(codigo=r.returncode, salida=r.stdout[-12000:], errores=r.stderr[-12000:])
    except subprocess.TimeoutExpired as e:
        data = {'timeout': True, 'salida': str(e.stdout)[-12000:], 'errores': str(e.stderr)[-12000:]}
    report[label] = data
    print(f'  Clientes: {data.get("clientes", {})}; respuesta: {data.get("respuesta_valida", False)}', flush=True)
    return bool(data.get('respuesta_valida')) and all(data.get('clientes', {}).values())

if not report['gazebo']:
    report['error'] = 'No se ve ningun gzserver en este entorno de procesos.'
    print(report['error'])
else:
    ok = run('prueba_terminal', os.environ.copy())
    selected = [p for p in report['gazebo']
                if p['entorno']['GAZEBO_MASTER_URI'] == os.environ.get('GAZEBO_MASTER_URI')]
    report['servidores_del_simulador'] = [p['pid'] for p in selected]
    if len(selected) == 1:
        server = selected[0]['entorno']
        report['diferencias'] = {k: {'terminal': os.environ.get(k), 'gazebo': server[k]}
                                for k in keys if os.environ.get(k) != server[k]}
        # No repite una prueba por rutas de biblioteca duplicadas; solo diferencias DDS.
        if any(k in report['diferencias'] for k in keys[:10]):
            env = os.environ.copy()
            for k, v in server.items():
                if v is None:
                    env.pop(k, None)
                else:
                    env[k] = v
            ok = run('prueba_entorno_gazebo', env) or ok
            print('Variables distintas:', ', '.join(report['diferencias']))
    if not ok:
        for local in ('1', '0'):
            env = os.environ.copy()
            env['FASTRTPS_DEFAULT_PROFILES_FILE'] = str(Path('base/fastdds_udp.xml').resolve())
            env['ROS_LOCALHOST_ONLY'] = local
            if run('prueba_udp_localhost_'+local, env):
                report['configuracion_que_funciona'] = {
                    'FASTRTPS_DEFAULT_PROFILES_FILE': env['FASTRTPS_DEFAULT_PROFILES_FILE'],
                    'ROS_LOCALHOST_ONLY': local}
                print('Comunicacion real comprobada con UDP y ROS_LOCALHOST_ONLY='+local)
                break
out.write_text(json.dumps(report, indent=2))
print(f'Diagnostico guardado en {out}')
PY
