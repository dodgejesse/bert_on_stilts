export GLUE_DIR=/home/jessedd/data/glue_data
export BERT_ALL_DIR=/home/jessedd/projects/bert_on_stilts/cache/bert_metadata



# to specify an experiment
export TASK=sst
SEED=4
DEBUG="debug_"
export OUTPUT_PATH=output/${TASK}/${DEBUG}seed_${SEED}/

if [ ${DEBUG} == "debug_" ]; then
    rm ${OUTPUT_PATH}*
fi


CUDA_VISIBLE_DEVICES=0 python glue/train.py \
    --task_name $TASK \
    --do_train --do_val --do_val_history \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_path /home/jessedd/data/bert/pretrained_weights/pytorch_model.bin \
    --bert_config_json_path /home/jessedd/data/bert/pretrained_weights/bert_config.json \
    --bert_load_mode model_only \
    --bert_save_mode model_all \
    --eval_init \
    --eval_during_train \
    --train_batch_size 4 \
    --learning_rate 2e-5 \
    --seed ${SEED} \
    --output_dir ${OUTPUT_PATH} \
    --train_examples_number 40 \
    --val_examples_number 40 \
