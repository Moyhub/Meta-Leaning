#!/bin/bash

DATAFN=1
DATATESTFN=1
GPUID=2

SEEDS=(42 667 128)
LR_META=(0.001)
SUPPORT_SIZE=(0)

for s in ${SEEDS[@]}; do
    for lm in ${LR_META[@]}; do
        for ss in ${SUPPORT_SIZE[@]}; do
            python main.py \
                --data_fn ${DATAFN} \
                --datatest_fn ${DATATESTFN} \
                --support_size ${ss} \
                --lr_meta ${lm} \
                --seed ${s} \
                --gpu_id ${GPUID} \
                --modeltype "cnn1d" \
                --do_eval
                # --do_train \
                # --do_eval
        done
    done
done


