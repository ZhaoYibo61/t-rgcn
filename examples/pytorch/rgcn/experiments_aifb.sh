#!/usr/bin/env bash

## Baselines without basis
#outp_file="rgcn_aifb_baseline.out"
#for i in {1..10}
#do
#    python entity_classify.py -d aifb --testing --gpu 0 --n-epochs 100 -r $i >> $outp_file
#done
#
# Basis comparison
#bases=( 20 40 60 80 100 )
#for base in ${bases[@]}
#do
#    outp_file="rgcn_aifb_paperbase_$base.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d aifb --testing --gpu 0 --n-epochs 50 -r $i --n-bases $base  >> $outp_file
#    done
#done
#
### Input embedding comparison
#embs=( 16 32 128 512 1024 )
#for emb in ${embs[@]}
#do
#    outp_file="rgcn_aifb_paperemb_$emb.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d aifb --testing --gpu 0 --n-epochs 50 -r $i --embedding --n-hidden $emb --n-bases -1 >> $outp_file
#    done
#done

#embs=( 8 16 24 32 48)
#for emb in ${embs[@]}
#do
#    outp_file="rgcn_aifb_hidden_$emb.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d aifb --testing --gpu 0 --n-epochs 100 -r $i --n-hidden $emb >> $outp_file
#    done
#done

#
## Attention comparison
#bases=( 1 10 30 50 70 80 91)
#for base in ${bases[@]}
#do
#    outp_file="rgcn_aifb_attention_base_$base.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d aifb --testing --gpu 0 --n-epochs 100 -r $i --n-bases $base -a  >> $outp_file
#    done
#done

## Tucker rank comparison
#ranks=( 1 10 30 50 70 80)
#for rank in ${ranks[@]}
#do
#    outp_file="rgcn_aifb_tucker_rank_$rank.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i --tucker -c $rank  >> $outp_file
#    done
#done

#ranks=( 30 50 100 200 )
#for rank in ${ranks[@]}
#do
#    outp_file="rgcn_aifb_tuckeRR_$rank.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d aifb --testing --gpu 0 --n-epochs 100 -r $i --tucker -c $rank  >> $outp_file
#    done
#done

# With Tucker
#embs=( 50 100 150 200 250 )
#for emb in ${embs[@]}
#do
#    outp_file="rgcn_aifb_tuckeR20new_$emb.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d aifb --testing --gpu 0 --n-epochs 100 -r $i --n-hidden $emb --tucker -c 20  >> $outp_file
#    done
#done

# With Tucker
bases=( 20 40 )
embs=( 16 32 )
for base in ${bases[@]}
do
    for emb in ${embs[@]}
    do
        outp_file="rgcn_aifb_papertucker_$base$emb.out"
        for i in {1..10}
        do
            python entity_classify.py -d aifb --testing --gpu 0 --n-epochs 100 -r $i --tucker -c $base,$emb,5 --patience 100  >> $outp_file
        done
    done
done