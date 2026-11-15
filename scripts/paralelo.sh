#!/usr/bin/env bash
# Parte C en simuladores sin ventana (5, 6, ...), sin tocar el Gazebo con ventana.
#   bash scripts/paralelo.sh recoger <modelo_dagger.pth> [pares=20] [procesos=4]
#       pares de trayectorias repartidos entre `procesos` simuladores
#   bash scripts/paralelo.sh rlhf <modelo_dagger.pth> [beta=5] [pasos=4000]
#       sim 5: RLHF beta>0 + estres | sim 6: RLHF beta=0 + estres | sim 7: estres de DAgger
# No pulses Ctrl+C: corta todos los procesos (los simuladores se cierran solos al salir).
set -eo pipefail
modo="${1:?recoger o rlhf}"; modelo="${2:?ruta del modelo DAgger}"
cd "$(dirname "$0")/.."
raiz="$PWD"
test -f "$modelo" || { echo "No existe $modelo"; exit 1; }
logs="$raiz/rlhf/results/logs/$modo"
mkdir -p "$logs"
if [[ "$modo" == "recoger" ]]; then
    pares="${3:-20}"; procesos="${4:-4}"
else
    beta="${3:-5}"; pasos="${4:-4000}"; procesos=3
    test -f rlhf/results/weights/recompensa_bt.pth || { echo "Falta recompensa_bt.pth: preferencias.py entrenar"; exit 1; }
fi
sims=$(seq 5 $((4+procesos)))
trabajos=()
limpiar() {
    for p in "${trabajos[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
    bash base/sim.sh parar $sims
    wait 2>/dev/null || true
}
trap limpiar EXIT

bash base/sim.sh parar $sims   # restos de corridas anteriores ocupan los puertos
for sim in $sims; do setsid bash base/sim.sh headless "$sim" > "$logs/sim$sim.log" 2>&1 & done
echo "Levantando simuladores $(echo $sims)..."
for sim in $sims; do
    until grep -q "Robot burger listo" "$logs/sim$sim.log"; do
        grep -q "No se pudo insertar\|no responde" "$logs/sim$sim.log" && { echo "Fallo el simulador $sim"; exit 1; }
        sleep 2
    done
done

en_sim() {  # en_sim <sim> <log> <comando...>: obstaculos propios + comando en ese simulador
    local sim="$1" log="$2"; shift 2
    (
        source base/ros.sh "$sim"
        python3 -u base/obstaculos.py --escenario cruce > "$logs/obstaculos$sim.log" 2>&1 &
        obst=$!
        trap 'kill -TERM $obst 2>/dev/null; wait $obst 2>/dev/null' EXIT
        until grep -q "Obstaculos en movimiento" "$logs/obstaculos$sim.log"; do sleep 1; done
        "$@" > "$log" 2>&1
    ) &
    trabajos+=($!)
}

if [[ "$modo" == "recoger" ]]; then
    por_proceso=$(( (pares + procesos - 1) / procesos ))
    for i in $(seq 0 $((procesos-1))); do
        desde=$((i*por_proceso)); hasta=$(( (i+1)*por_proceso )); (( hasta > pares )) && hasta=$pares
        (( desde >= hasta )) && continue
        en_sim $((5+i)) "$logs/recoger$((5+i)).log" python3 -u rlhf/src/preferencias.py recoger \
            --modelo "$modelo" --pares "$pares" --desde "$desde" --hasta "$hasta"
        echo "Simulador $((5+i)): pares $desde..$((hasta-1))"
    done
else
    en_sim 5 "$logs/rlhf_beta.log" bash -c "python3 -u rlhf/src/rlhf.py --corrida beta --beta $beta --pasos $pasos --referencia '$modelo' && python3 -u rlhf/src/estres.py --politica beta --referencia '$modelo'"
    en_sim 6 "$logs/rlhf_sin_beta.log" bash -c "python3 -u rlhf/src/rlhf.py --corrida sin_beta --beta 0 --pasos $pasos --referencia '$modelo' && python3 -u rlhf/src/estres.py --politica sin_beta --referencia '$modelo'"
    en_sim 7 "$logs/estres_dagger.log" python3 -u rlhf/src/estres.py --politica dagger --referencia "$modelo"
    echo "En marcha. Avance: tail -f $logs/rlhf_beta.log $logs/rlhf_sin_beta.log"
fi

estado=0
for t in "${trabajos[@]}"; do wait "$t" || estado=1; done
if [[ "$modo" == "recoger" ]]; then
    echo "Trayectorias: $(ls rlhf/results/data/trayectorias/*.npz 2>/dev/null | wc -l) de $((2*pares))"
else
    partes="rlhf/results/logs/estres_partes"
    if [[ -f $partes/dagger.csv && -f $partes/beta.csv && -f $partes/sin_beta.csv ]]; then
        { head -1 "$partes/dagger.csv"; for p in dagger beta sin_beta; do tail -n +2 "$partes/$p.csv"; done; } \
            > rlhf/results/logs/estres.csv
        column -s, -t < rlhf/results/logs/estres.csv
    fi
fi
(( estado )) && echo "Algun proceso fallo: revisa $logs/*.log"
exit $estado
