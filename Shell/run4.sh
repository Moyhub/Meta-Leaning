#!/bin/bash

DATAFN=4
GPUID=0

SEEDS=(42 667 128)
LR_META=(0.001 0.0001 0.0005)
SUPPORT_SIZE=(0)
for s in ${SEEDS[@]}; do
    for lm in ${LR_META[@]}; do
        for ss in ${SUPPORT_SIZE[@]}; do
            python main.py \
                --data_fn ${DATAFN} \
                --support_size ${ss} \
                --lr_meta ${lm} \
                --seed ${s} \
                --gpu_id ${GPUID}
        done
    done
done

SEEDS=(42 667 128)
EPOCHS=(5 10)
LR_META=(0.001 0.0001 0.0005)
SUPPORT_SIZE=(2 5 10)
LR_INNER=(0.0001)
INNER_STEPS=(1 2)
for s in ${SEEDS[@]}; do
    for e in ${EPOCHS[@]}; do
        for ss in ${SUPPORT_SIZE[@]}; do
            for lm in ${LR_META[@]}; do
                for li in ${LR_INNER[@]}; do
                    for is in ${INNER_STEPS[@]}; do
                        python main.py \
                            --data_fn ${DATAFN} \
                            --support_size ${ss} \
                            --inner_steps ${is} \
                            --lr_meta ${lm} \
                            --lr_inner ${li} \
                            --seed ${s} \
                            --gpu_id ${GPUID} \
                            --n_epochs ${e}
                    done
                done
            done
        done
    done
done