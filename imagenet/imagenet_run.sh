BS=256
SEED=100

CUDA_VISIBLE_DEVICES=0 python optim_ensemble.py \
    --dataset ImageNet \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 1e-4 \
    --conformal_erm_p 0.5 \
    --name p_0.5_lamlr_1e-4 \
    --seed "$SEED"