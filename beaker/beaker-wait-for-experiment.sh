#!/bin/bash

ex=${@?"Need an experiment id"}

echo -n "Waiting for all tasks in experiment $ex to succeed..."

get_statuses() {
   echo $(beaker experiment inspect $ex | jq -r '.[].nodes[].status' | sort | uniq | tr '\n' '-')
}

while true; do
  statuses="$(get_statuses)"


  if ! [[ ${statuses} == *"running"* ]]; then
      echo "done"
  
      if [[ "$statuses" == "succeeded-" ]]; then
	  echo "all experiments succeeded."
	  exit 0
      else
	  echo "something didn't succeed."
	  exit 1
      fi
  fi
  sleep 1
  echo -n .
done
