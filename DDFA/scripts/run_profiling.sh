#!/bin/bash

ckpt="$1"

for metric in profile time
do
    bash scripts/test.sh $ckpt --model.$metric True
done
