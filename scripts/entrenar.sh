#!/usr/bin/env bash
# Entrena y evalua un metodo de la Parte A en un simulador ya levantado (bash base/sim.sh ...).
#   bash scripts/entrenar.sh <a2c|ppo> <cruce|facil> [sim=1] [pasos=10000]
# Lanza los obstaculos, entrena (10 validaciones), evalua actor_base.pth en 5 episodios y retira los obstaculos.
set -eo pipefail
metodo="${1:?a2c o ppo}"; escenario="${2:?cruce o facil}"; pasos="${4:-10000}"
source "$(dirname "$0")/../base/ros.sh" "${3:-1}"
cd "$LAB4_RAIZ"
logs="$metodo/results/logs/$escenario"
mkdir -p "$logs"
python3 -u base/obstaculos.py --escenario "$escenario" > "$logs/obstaculos.log" 2>&1 &
obst=$!
trap 'kill -TERM $obst 2>/dev/null; wait $obst 2>/dev/null' EXIT
until grep -q "Obstaculos en movimiento" "$logs/obstaculos.log"; do
    kill -0 $obst 2>/dev/null || { cat "$logs/obstaculos.log"; exit 1; }
    sleep 1
done
python3 -u "$metodo/src/$metodo.py" --corrida "$escenario" --escenario "$escenario" --pasos "$pasos" \
    --validar-cada "$((pasos/10))" 2>&1 | tee "$logs/entrenamiento.log"
python3 -u base/evaluar.py --pesos "$metodo/results/weights/$escenario/actor_base.pth" \
    --escenario "$escenario" --episodios 5 --salida "$logs/eval" 2>&1 | tee "$logs/evaluacion.log"
