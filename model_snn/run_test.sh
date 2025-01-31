#!/bin/bash -e
BASEDIR="../data/log"
LOG_NAME=$(date +"%m-%d_%H-%M-%S")
LOG_OUTDIR="$BASEDIR/test_$LOG_NAME"
echo $LOG_OUTDIR
mkdir $LOG_OUTDIR
mkdir "$LOG_OUTDIR/code"
cp -r ./* "$LOG_OUTDIR/code"

# nohup python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 56565 test.py \
nohup python -u test.py \
    --visible_gpus         "0" \
    --data_dir             "/home/argha/sample_data" \
    --output_dir           "$LOG_OUTDIR" \
    --model_dir            "/home/argha/github/HumanPoseTracking_SNN/data/log/12-24_17-39-47/model_snn.pth" \
    --save_vis             0 \
    --save_attention_map   0 \
    --smpl_dir             "/home/argha/smpl_model/smpl/models/SMPL_MALE.pkl" \
    --channel              4 \
    --num_frames           8 \
    --img_size             256 \
    --max_gap              4 \
    --use_mmhpsd           1 \
    --use_mmhpsd_synthesis 0 \
    --use_h36m             0 \
    --use_amass            0 \
    --use_phspd            0 \
    --use_geodesic_loss    1 \
    --use_amp              0 \
    --backbone             "sew_resnet18" \
    --neuron               "ParametricLIFNode" \
    --surrogate            "ATan" \
    --cnf                  "ADD" \
    --detach_reset         0 \
    --hard_reset           0 \
    --drop_prob            0.1 \
    --n_layers             0 \
    --n_head               0 \
    --d_hidden             1024 \
    --use_rnn              0 \
    --use_recursive        0 \
    --use_transformer      1 \
    --trans_loss           50 \
    --theta_loss           10 \
    --beta_loss            1 \
    --joints3d_loss        1 \
    --joints2d_loss        10 \
    --batch_size           1 \
    --epochs               20 \
    --lr_scheduler         "CosineAnnealingLR" \
    --lr                   0.01 \
    --lr_regressor         0.0001 \
    --lr_decay_rate        1 \
    --lr_decay_step        21 \
> "$LOG_OUTDIR/printlog.txt" 2>&1 &
