EPOCHS=10000
RECAL_RATIO=0.1
TEST_RATIO=0.1
NAME=cqr_nolog_sgdplateau_hingedual

DATASET=meps_19
SEED=100

CUDA_VISIBLE_DEVICES=0 python conformal_finetuning.py \
    --dataset "$DATASET" \
    --mode quantile \
    --network_mode neural_net \
    --batch_size 1024 \
    --epochs "$EPOCHS" --scheduler plateau --disp_per_epoch 100 \
    --optimizer sgd --decay_max_times 2 \
    --no_log_transform \
    --cal_ratio 0.1 --recal_ratio "$RECAL_RATIO" --test_ratio "$TEST_RATIO" \
    --name "$NAME" \
    --conformal_erm_lr 0.01 \
    --conformal_erm_lam_update loss \
    --seed "$SEED"