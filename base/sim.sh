#!/usr/bin/env bash
# Simuladores Gazebo (mundo turtlebot3_dqn_stage1) con el TurtleBot3 burger.
#   bash base/sim.sh gui              simulador 1 con ventana (teleoperacion, video)
#   bash base/sim.sh headless 5       simulador 5 sin ventana (entrenamientos en paralelo)
#   bash base/sim.sh parar 5 6 7      cierra por completo esos simuladores
# Si el spawn del launch falla por timeout, el robot se inserta solo.
modo="${1:-gui}"

if [[ "$modo" == "parar" ]]; then
    # Se identifican por su entorno (GAZEBO_MASTER_URI + ROS_DOMAIN_ID), no por el nombre del comando.
    shift
    for sim in "$@"; do
        uri="GAZEBO_MASTER_URI=http://localhost:$((11344 + sim))"; dominio="ROS_DOMAIN_ID=$((29 + sim))"
        for p in $(pgrep -u "$USER" -f "gzserver|sim.sh headless|obstaculos.py|ros2 launch|spawn_entity"); do
            [[ "$p" == "$$" ]] && continue
            env=$(tr '\0' '\n' 2>/dev/null < /proc/$p/environ) || continue
            grep -qx "$uri" <<< "$env" && grep -qx "$dominio" <<< "$env" && kill -TERM "$p" 2>/dev/null
        done
    done
    sleep 2
    for sim in "$@"; do
        uri="GAZEBO_MASTER_URI=http://localhost:$((11344 + sim))"
        for p in $(pgrep -x gzserver); do
            tr '\0' '\n' 2>/dev/null < /proc/$p/environ | grep -qx "$uri" && kill -KILL "$p" 2>/dev/null
        done
    done
    exit 0
fi

source "$(dirname "$0")/ros.sh" "${2:-1}"
echo "Simulador $LAB4_SIM: ROS_DOMAIN_ID=$ROS_DOMAIN_ID GAZEBO_MASTER_URI=$GAZEBO_MASTER_URI"
TB3="$(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo"

esperar_factory() {
    local hasta=$((SECONDS + 90))
    while (( SECONDS < hasta )); do
        if timeout 6 ros2 service list --no-daemon --spin-time 2 2>/dev/null | grep -qx /spawn_entity; then
            return 0
        fi
        if [[ -n "${servidor:-}" ]] && ! kill -0 "$servidor" 2>/dev/null; then
            echo "[sim] El servidor termino antes de ofrecer /spawn_entity." >&2
            return 1
        fi
        sleep 1
    done
    echo "[sim] Gazebo no responde: no aparece /spawn_entity en 90 s." >&2
    return 1
}

asegurar_robot() {
    esperar_factory || return 1
    sleep 15   # deja terminar el spawn del launch
    for intento in 1 2 3; do
        if timeout 10 ros2 service call /get_model_list gazebo_msgs/srv/GetModelList 2>/dev/null | grep -q "'burger'"; then
            echo "[sim] Robot burger listo."; return 0
        fi
        echo "[sim] Falta el robot (intento $intento): insertandolo..."
        timeout 60 ros2 run gazebo_ros spawn_entity.py -entity burger -file "$TB3/models/turtlebot3_burger/model.sdf" \
            -x 0 -y 0 -z 0.01 2>&1 | grep -E "Successfully|rror" || true
        sleep 3
    done
    echo "[sim] No se pudo insertar el robot."; return 1
}

asegurar_robot &
vigia=$!
trap 'kill $vigia $servidor 2>/dev/null' EXIT

if [[ "$modo" == "gui" ]]; then
    ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage1.launch.py
else
    ros2 launch gazebo_ros gzserver.launch.py world:="$TB3/worlds/turtlebot3_dqn_stage1.world" \
        init:=true factory:=true server_required:=true verbose:=true &
    servidor=$!
    esperar_factory || exit 1
    ros2 launch turtlebot3_gazebo spawn_turtlebot3.launch.py x_pose:=0.0 y_pose:=0.0
    wait $servidor
fi
