#!/bin/bash

export PYTHONPATH="$(pwd)"


level='level2'
patches='patches_v1'
fold='1'

name="rip_$patches-$level"
dirname="/data2/data2/zewei/exp/RipData/DeepLabV3/$patches/$level/CV5-$fold"

echo "output" $dirname

if [ ! -d "$dirname" ]
then
    echo "$dirname doesn't exist. Creating now"
    mkdir $dirname
    echo "File created"
else
    echo "$dirname exists"
fi

python -u train.py \
     --backbone resnet \
     --lr 0.01 \
     --workers 8 \
     --epochs 40 \
     --batch_size 4 \
     --gpus 1 \
     --checkname deeplab-resnet \
     --eval_interval 1 \
     --dataset rip \
     --rip_mode $patches-$level \
     --exp_root $dirname \
     "$@" \
     2>&1 | tee -a $dirname/$name.log

