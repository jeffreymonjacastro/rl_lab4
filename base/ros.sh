# Entorno ROS 2 de un simulador concreto.   Uso: source base/ros.sh [sim]
#   sim 1 = Gazebo con ventana; 2..N = simuladores sin ventana aislados (dominio ROS y puerto propios).
source /opt/ros/humble/setup.bash
[[ -f "$HOME/rl_ws/install/setup.bash" ]] && source "$HOME/rl_ws/install/setup.bash"
source /usr/share/gazebo/setup.sh
export LAB4_RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Misma implementacion y descubrimiento estandar en Gazebo, obstaculos y entrenamiento.
# LAB4_ROS_LOCAL=0 conserva la configuracion DDS externa para uso en red.
if [[ "${LAB4_ROS_LOCAL:-1}" == "1" ]]; then
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    # Usa el descubrimiento estandar en las interfaces de WSL.
    # Servidor y clientes deben cargar el mismo valor; el dominio separa simuladores.
    export ROS_LOCALHOST_ONLY=0
    # Limpia el perfil experimental tambien en terminales donde ya se cargo.
    # Usa los transportes y QoS estandar de rmw_fastrtps_cpp.
    unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
    unset RMW_FASTRTPS_USE_QOS_FROM_XML ROS_DISCOVERY_SERVER
fi
export LAB4_SIM="${1:-${LAB4_SIM:-1}}"
export ROS_DOMAIN_ID=$((29 + LAB4_SIM))
export GAZEBO_MASTER_URI="http://localhost:$((11344 + LAB4_SIM))"
# Todos los simuladores son locales: evita que WSL anuncie una interfaz virtual
# que Gazebo no pueda alcanzar. Se puede indicar otra IP antes de cargar ros.sh.
export GAZEBO_IP="${GAZEBO_IP:-127.0.0.1}"
export TURTLEBOT3_MODEL=burger
