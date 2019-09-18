cd mnist-example

EXPERIMENT_IDS=""
for SEED in {1..99}; do
    EXPERIMENT_IDS="${EXPERIMENT_IDS} `RANDOM_SEED=${SEED} beaker experiment create -f spec.yml -q`"
    #EXPERIMENT_IDS="${EXPERIMENT_IDS} `echo $SEED`"
done

echo ""
echo "experiment ids: ${EXPERIMENT_IDS}"

cd ..

bash beaker-wait-for-experiment.sh ${EXPERIMENT_IDS}

echo "next, going to get logs for the following experiments:"
echo "${EXPERIMENT_IDS}"

for CUR_ID in ${EXPERIMENT_IDS}; do
    bash beaker-get-logs.sh ${CUR_ID}
    sleep 3
done

echo "\n"
echo "the experiment ids:"
echo "${EXPERIMENT_IDS}"

sleep 10
tail -n 1 output/*.log | grep accuracy | awk '{print $4 $8}' > output/train_performance.txt
