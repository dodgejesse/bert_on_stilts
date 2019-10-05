#!/bin/bash

ex=$1
TASK=$2
for t in $(beaker experiment inspect $ex | jq -r '.[0].nodes[].taskId'); do

    echo "Saving results dataset for experiment $ex task ${TASK} to ${TASK}/$out"
    
    RESULT_ID=`beaker experiment inspect ${ex} | jq -r '.[0].nodes[].resultId'`
    beaker dataset fetch -o output/${TASK}/ ${RESULT_ID}
done
