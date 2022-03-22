BS=64

python optim_ensemble.py \
    --dataset ImageNetV2 \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 1e-2 \
    --conformal_erm_p 1.0 \
    --name p_1.0_lamlr_1e-2 \
    --seed 100

python optim_ensemble.py \
    --dataset ImageNetV2 \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 1e-3 \
    --conformal_erm_p 1.0 \
    --name p_1.0_lamlr_1e-3 \
    --seed 100

python optim_ensemble.py \
    --dataset ImageNetV2 \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 1e-4 \
    --conformal_erm_p 1.0 \
    --name p_1.0_lamlr_1e-4 \
    --seed 100

python optim_ensemble.py \
    --dataset ImageNetV2 \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 0.0 \
    --conformal_erm_p 1.0 \
    --name p_1.0_lamlr_0.0 \
    --seed 100


# p=0.5
python optim_ensemble.py \
    --dataset ImageNetV2 \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 1e-2 \
    --conformal_erm_p 0.5 \
    --name p_0.5_lamlr_1e-2 \
    --seed 100

python optim_ensemble.py \
    --dataset ImageNetV2 \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 1e-3 \
    --conformal_erm_p 0.5 \
    --name p_0.5_lamlr_1e-3 \
    --seed 100

python optim_ensemble.py \
    --dataset ImageNetV2 \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 1e-4 \
    --conformal_erm_p 0.5 \
    --name p_0.5_lamlr_1e-4 \
    --seed 100

python optim_ensemble.py \
    --dataset ImageNetV2 \
    --batch_size "$BS" \
    --conformal_erm_epochs 500 \
    --conformal_erm_lam_lr 0.0 \
    --conformal_erm_p 0.5 \
    --name p_0.5_lamlr_0.0 \
    --seed 100