
TASK=cola


#for PAIR in "1 6" "1 8" "2 1" "2 10" "3 1" "3 3" "4 2" "4 6" "5 9" "6 3" "7 3" "7 4" "7 10" "8 2" "8 5" "9 3" "9 5" "9 8" "10 1" "10 4" "10 5" "10 10"; do
#    INIT_SEED=$(echo $PAIR | cut -f1 -d " ")
#    DATA_SEED=$(echo $PAIR | cut -f2 -d " ")



EXPERIMENT_IDS=""
for INIT_SEED in {1..20}; do
    for DATA_SEED in {21..25}; do
	EXPERIMENT_IDS="${EXPERIMENT_IDS} `INIT_SEED=${INIT_SEED} DATA_SEED=${DATA_SEED} TASK=${TASK} beaker experiment create -f spec.yml -q`"
    done
done

for INIT_SEED in {21..25}; do
    for DATA_SEED in {1..25}; do
	EXPERIMENT_IDS="${EXPERIMENT_IDS} `INIT_SEED=${INIT_SEED} DATA_SEED=${DATA_SEED} TASK=${TASK} beaker experiment create -f spec.yml -q`"
    done
done


echo ""
echo "experiment ids: ${EXPERIMENT_IDS}"
echo "experiment ids: ${EXPERIMENT_IDS}" > cur_experiment_ids.txt


bash beaker-wait-for-experiment.sh ${EXPERIMENT_IDS}

echo "next, going to get logs for the following experiments:"
echo "${EXPERIMENT_IDS}"

for CUR_ID in ${EXPERIMENT_IDS}; do
    bash beaker-get-results-dataset.sh ${CUR_ID} ${TASK}
    sleep 3
done

echo "\n"
echo "the experiment ids:"
echo "${EXPERIMENT_IDS}"

sleep 10
#tail -n 1 output/*.log | grep accuracy | awk '{print $4 $8}' > output/train_performance.txt
