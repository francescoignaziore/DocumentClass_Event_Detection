#!/usr/bin/python

python run_glue_d.py \
--model_name_or_path $MODEL \
--run_name $TASK \
--train_file train_ds_full.json \
--validation_file val_ds_full.json \
--do_train \ 
--do_eval \
--per_device_train_batch_size 16 \
--learning_rate 2e-5 \
--num_train_epochs 4 \
--logging_steps 500 \
--save_strategy no \
--save_total_limit 1 \
--output_dir $SCRATCH/test_runs/$TASK \
--fp16
