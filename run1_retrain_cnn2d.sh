#!/bin/bash

DATAFN=3
DATATESTFN=3
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
                --modeltype "cnn2d" \
                --do_eval
                #--do_train \
        done
    done
done

SEEDS=(42 667 128)
LR_META=(0.001)
SUPPORT_SIZE=(2 5)
LR_INNER=(0.0001 0.001)
INNER_STEPS=(2)

for s in ${SEEDS[@]}; do
   for lm in ${LR_META[@]}; do
       for ss in ${SUPPORT_SIZE[@]}; do
           for li in ${LR_INNER[@]}; do
               for is in ${INNER_STEPS[@]}; do
                   python main.py \
                       --data_fn ${DATAFN} \
                       --datatest_fn ${DATATESTFN} \
                       --support_size ${ss} \
                       --inner_steps ${is} \
                       --lr_meta ${lm} \
                       --lr_inner ${li} \
                       --seed ${s} \
                       --gpu_id ${GPUID} \
                       --modeltype "cnn2d" \
                       --do_eval
                    #    --do_train  \
                    #    --do_eval 
               done
           done
       done
   done
done



