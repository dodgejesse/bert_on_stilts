#!/bin/bash
# taken from https://github.com/allenai/aimichal/blob/master/beaker-wait-for-experiment.sh

EXPERIMENT_IDS=${@?"Need an experiment id"}

echo "Waiting for all experiments to finish in: ${EXPERIMENT_IDS}"

get_statuses() {
   echo $(beaker experiment inspect $1 | jq -r '.[].nodes[].status' | sort | uniq)
}
# used to be
# echo $(beaker experiment inspect $1 | jq -r '.[].nodes[].status' | sort | uniq | tr '\n' '-')


# all statuses listed: https://github.com/beaker/client/blob/master/api/task_status.go
for EX_ID in ${EXPERIMENT_IDS}; do
    echo -n "Waiting for experiment ${EX_ID}..."
    while true; do
	statuses="$(get_statuses ${EX_ID})"

	if [[ "$statuses" == "succeeded" ]]; then
	    echo "experiment ${EX_ID} succeeded."
	    break
	fi
	if [[ "$statuses" == "failed" ]] || [[ "$statuses" == "stopped" ]] || [[ "$statuses" == "skipped" ]]; then
	    echo "experiment ${EX_ID} $statuses"
	    break
	fi
	sleep 1
	echo -n .
    done
done
