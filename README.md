# syn_ant

 ```sh
# example of inference
# model should be a checkpoint from one of the fine-tuning methods below
echo 'good bad evil terrorist freedom_fighter white black' |
 awk '{for(i=1;i<=NF;i++) for(j=1;j<=NF;j++) print $i "\t" $j}' |
tr _ ' ' |
 python sentiment4_inference.py --model $model 


# syn/ant classification
# fine-tunes a model
# pos should be one of: adj, noun, verb, fallows
# model should be something like bert-base-uncased (or a checkpoint)
python sentiment4.py --checkpoint_dir $res/checkpoints --pos $pos --epochs 400 --pretrained $model


# VAD regression
# fine-tunes a model

# model should be something like bert-base-uncased (or a checkpoint)
python fine_tune_VAD_pairs.py  --pretrained $model --epochs 300 --checkpoint_dir $res/checkpoints --VAD_path syn_ant/datasets/datasets_VAD/simple/VAD.simple.1000k
```

