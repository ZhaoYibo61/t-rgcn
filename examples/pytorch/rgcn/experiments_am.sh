#!/usr/bin/env bash

# Baselines without basis
outp_file="rgcn_am_baseline.out"
for i in {1..10}
do
    python entity_classify.py -d am --testing --gpu 0 --n-epochs 100 -r $i >> $outp_file
done

# Basis comparison
bases=( 1 10 30 50 70 80)
for base in ${bases[@]}
do
    outp_file="rgcn_am_base_$base.out"
    for i in {1..10}
    do
        python entity_classify.py -d am --testing --gpu 0 --n-epochs 100 -r $i --n-bases $base  >> $outp_file
    done
done

# Tucker rank comparison
ranks=( 1 10 30 50 70 80)
for rank in ${ranks[@]}
do
    outp_file="rgcn_am_tucker_rank_$base.out"
    for i in {1..10}
    do
        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i --tucker --core_t $rank  >> $outp_file
    done
done

# Input embedding comparison
embs=( 8 16 24 32 48)
for emb in ${embs[@]}
do
    outp_file="rgcn_am_emb_$emb.out"
    for i in {1..10}
    do
        python entity_classify.py -d am --testing --gpu 0 --n-epochs 100 -r $i --embedding --n-hidden $emb >> $outp_file
    done
done

# Attention comparison
bases=( 1 10 30 50 70 80 91)
for base in ${bases[@]}
do
    outp_file="rgcn_am_attention_base_$base.out"
    for i in {1..10}
    do
        python entity_classify.py -d am --testing --gpu 0 --n-epochs 100 -r $i --n-bases $base -a  >> $outp_file
    done
done

