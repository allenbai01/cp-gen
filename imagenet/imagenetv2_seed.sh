BS=64

# 100 200 300 400
for SEED in 500 600 700 800
do
python optim_ensemble.py \
    --dataset ImageNetV2 \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 1e-4 \
    --conformal_erm_p 0.5 \
    --name p_0.5_lamlr_1e-4 \
    --seed "$SEED"

python optim_ensemble.py \
    --dataset ImageNetV2 \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 1e-4 \
    --conformal_erm_p 1.0 \
    --name p_1.0_lamlr_1e-4 \
    --seed "$SEED"
done