#!/usr/bin/env bash

## Baselines without basis
#outp_file="rgcn_mutag_baseline.out"
#for i in {1..10}
#do
#    python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i >> $outp_file
#done
#
# Basis comparison
#bases=( 10 20 30 40 50 )
#for base in ${bases[@]}
#do
#    outp_file="rgcn_mutag_paperbase_$base.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 50 -r $i --n-bases $base  >> $outp_file
#    done
#done
#
# Input embedding comparison
embs=( 16 32 128 512 1024 )
for emb in ${embs[@]}
do
    outp_file="rgcn_mutag_paperemb_$emb.out"
    for i in {1..10}
    do
        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 50 -r $i --embedding --n-hidden $emb --n-bases -1 >> $outp_file
    done
done

#embs=( 8 16 24 32 48)
#for emb in ${embs[@]}
#do
#    outp_file="rgcn_mutag_hidden_$emb.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i --n-hidden $emb >> $outp_file
#    done
#done



#
## Attention comparison
#bases=( 1 10 20 30 47)
#for base in ${bases[@]}
#do
#    outp_file="rgcn_mutag_attention_base_$base.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i --n-bases $base -a  >> $outp_file
#    done
#done

## Tucker rank comparisons
#ranks=( 1 10 20 30 40)
#for rank in ${ranks[@]}
#do
#    outp_file="rgcn_mutag_tucker_rank_$rank.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i --tucker -c $rank  >> $outp_file
#    done
#done


## Tucker rank comparisons
#ranks=( 30 50 100 200 )
#for rank in ${ranks[@]}
#do
#    outp_file="rgcn_mutag_tuckeRR_$rank.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i --tucker -c $rank  >> $outp_file
#    done
#done

# Tucker Rank percentage comparisons
#rankp=( 10 50 100 150 200 )
#for rank in ${rankp[@]}
#do
#    outp_file="rgcn_mutag_tucker_rankp_$rank.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i --tucker --rank-per $rank  >> $outp_file
#    done
#done


## Input embedding comparison
#embs=( 300 500 700 1000 )
#for emb in ${embs[@]}
#do
#    outp_file="rgcn_mutag_embcomp_$emb.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i --embedding --n-hidden $emb >> $outp_file
#    done
#done


## Tucker Rank percentage comparisons
#rankp=( 10 50 100 150 200 )
#for rank in ${rankp[@]}
#do
#    outp_file="rgcn_mutag_tucker_rankp_reg_$rank.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i --tucker --rank-per $rank --orthogonal_reg  >> $outp_file
#    done
#done

# With Tucker
#embs=( 50 100 150 200 250 )
#for emb in ${embs[@]}
#do
#    outp_file="rgcn_mutag_tuckeR20new_$emb.out"
#    for i in {1..10}
#    do
#        python entity_classify.py -d mutag --testing --gpu 0 --n-epochs 100 -r $i --n-hidden $emb --tucker -c 20 >> $outp_file
#    done
#done
