# nclients=(2 4 8 16 32 64)
# nrounds=(0 10 20 40 80 160)
# nrounds=(10 20 40 80 160)
# nrounds=(1 4 16 32)

nclients=(4 8 16 32 64)
nrounds=(32)

DS="toniot"
# DS="botiot"
output_path="logs/scci2023"

for ROUNDS in ${nrounds[*]}; do
    echo "######################################################################"
    echo "$ROUNDS Rounds"
    for NC in ${nclients[*]}; do
        echo "Clients: $NC"
        exp_name="experiment_${ROUNDS}rounds_${NC}clients"
        exp_path="${output_path}/${exp_name}.pickle"
        exp_path_log="${output_path}/${exp_name}.txt"
        python3 experiment.py -f ../bot-iot-exploratory/Train_Test_Network.csv -nc $NC -d $DS -nr $ROUNDS -vs 1 -vc 1 -en $exp_name -o $exp_path &> $exp_path_log
    done
    echo "######################################################################"
done

# for NC in ${nclients[*]}; do
#   echo "######################################################################"
#   echo "$NC Clients"
#   python3 experiment.py -f ../bot-iot-exploratory/Train_Test_Network.csv -d toniot -nr 0 -nes 1 -vs 1 -vc 1 -en experiment1
#   echo "######################################################################"
# done
