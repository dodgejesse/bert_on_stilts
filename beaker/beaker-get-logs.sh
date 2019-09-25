#!/bin/bash

ex=$1
TASK=$2
for t in $(beaker experiment inspect $ex | jq -r '.[0].nodes[].taskId'); do
  out="$t.log"
  echo "Saving logs for experiment $ex task $t to ${TASK}/$out"
  curl -L http://beaker.org/api/v3/tasks/$t/logs -H 'cookie: User-Token=token-2hWylZs0ssKdbn4f;' > output/${TASK}/$out
done
