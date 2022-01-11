# Project

Based on sequence modeling toolkit Fairseq (https://github.com/pytorch/fairseq)

## Requirements
- PyTorch version >= 1.4.0
- Python version >= 3.6

Data : NUCLE and Lang-8 (https://disk.pku.edu.cn:443/link/C80DE4D17DD784632FB9A5C31F88AEC7
expiry date：2022-02-28 23:59)

## Preprocess

```
python preprocess_graph.py --trainpref edge-data/train --validpref edge-data/valid \
--testpref edge-data/test --source-lang src --target-lang tgt --destdir data/bin \
--nwordssrc 50000 --workers 5 --edgedict edge-data/dict.edge.txt \
--task translation_with_graph_attention_with_copy
```
```
python process_graph_copy.py --testpref edge-data/test --source-lang src --target-lang tgt \
--destdir data/bin-copy  --nwordssrc 50000 --workers 5 --task translation_with_graph_attention_with_copy \
--edgedict edge-data/dict.edge.txt --srcdict data/bin/dict.src.txt \
--tgtdict data/bin/dict.tgt.txt --dataset-impl raw
```

## Train

```
ARCH=transformer_concat_with_graph_copy_gigaword_big2

CUDA_VISIBLE_DEVICES=0 python train.py data/bin \
  -a $ARCH --optimizer adam --lr 0.0001 -s src -t tgt \
  --dropout 0.1 --max-tokens 2048 \
  --share-decoder-input-output-embed \
  --task translation_with_graph_attention_with_copy \
  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/$ARCH \
  --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 --criterion cross_entropy_copy --update-freq 2
```

## Test
```
CUDA_VISIBLE_DEVICES=0  python generate.py data/bin-copy \
--task $ARCH  \
--path  checkpoints/$ARCH/checkpoint_best.pt \
--batch-size 128 --beam 5 --lenpen 1.2 --replace-unk --raw-text \
> output/$ARCH/conll14st-test.tok.trg 
```

## Evaluation
```
grep ^H output/$ARCH/conll14st-test.tok.trg | sort -n -k 2 -t '-' | cut -f 1,3 > output/$ARCH/conll14st-test.tmp

python output/get_output.py output/$ARCH/conll14st-test.tmp output/$ARCH/conll14st-test.out

python m2scorer/scripts/m2scorer.py  output/$ARCH/conll14st-test.out \
data/conll14st-test.m2
```

