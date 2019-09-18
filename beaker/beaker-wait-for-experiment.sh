#!/bin/bash

ex=${1?"Need an experiment id"}

echo -n "Waiting for all tasks in experiment $ex to succeed..."

get_statuses() {
   echo $(beaker experiment inspect $ex | jq -r '.[].nodes[].status' | sort | uniq | tr '\n' '-')
}

while true; do
  statuses="$(get_statuses)"
  if [[ "$statuses" == "succeeded-" ]]; then
    echo done
    exit 0
  fi
  if [[ "$statuses" =~ "failed" ]]; then
    echo something failed 
    exit 1
  fi
  sleep 1
  echo -n .
done
