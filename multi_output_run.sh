EPOCHS=1000
RECAL_RATIO=0.1
TEST_RATIO=0.1
ALPHA=0.1

DATASET=cartpole
NAME=lam_loss_cerm_lr0.01_lamlr0.01_alpha_"$ALPHA"
SEED=100

CUDA_VISIBLE_DEVICES=0 python multi_output.py \
    --dataset "$DATASET" \
    --alpha "$ALPHA" \
    --epochs "$EPOCHS" --decay_per_epoch 500 --disp_per_epoch 5 \
    --conformal_erm_lam_update loss \
    --conformal_erm_lr 0.01 --conformal_erm_lam_lr 0.01 \
    --cal_ratio 0.1 --recal_ratio "$RECAL_RATIO" --test_ratio "$TEST_RATIO" \
    --name "$NAME" \
    --seed "$SEED"