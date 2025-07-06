
config="configs/smpl.json"
gpu_id=0
motion=1.0


python predict.py \
    --config $config \
    --gpu_id $gpu_id \
    --motion $motion