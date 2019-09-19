export GLUE_DIR=/data/
export BERT_ALL_DIR=bert_on_stilts/cache/bert_metadata



# to specify an experiment
export TASK=sst
SEED=${RANDOM_SEED}
DEBUG="debug_"
export OUTPUT_PATH=output/sst/${DEBUG}seed_${SEED}/

if [ ${DEBUG} == "debug_" ]; then
    rm ${OUTPUT_PATH}*
fi


python bert_on_stilts/glue/train.py \
    --task_name $TASK \
    --do_train --do_val --do_val_history \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_path /pretrained_weights/pytorch_model.bin \
    --bert_config_json_path /pretrained_weights/bert_config.json \
    --bert_load_mode model_only \
    --bert_save_mode model_all \
    --eval_init \
    --train_batch_size ${BATCH_SIZE} \
    --learning_rate 2e-5 \
    --seed ${SEED} \
    --output_dir ${OUTPUT_PATH} \
    --train_examples_number 5000 \
