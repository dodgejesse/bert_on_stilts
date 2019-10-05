
TASK=mrpc

EXPERIMENT_IDS=""
for INIT_SEED in {1..1}; do
    for DATA_SEED in {1..1}; do
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
