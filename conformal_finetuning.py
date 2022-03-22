import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os, sys
import numpy as np
import time
import argparse
import copy
np.warnings.filterwarnings('ignore')

from cqr.datasets import datasets # requires scikit-learn <= 0.19.2
from models import MLP, TwoSidedPinballLoss
from conformal import compute_critical_score, evaluate_coverage_length
from utils import AverageMeter, FreezeReps
from data_utils import get_epochs, lr_power_decay
import wandb


def train(epoch, net, optimizer, criterion, loader,
          verbose=False, disp_per_batch=10):
    net.train()
    if verbose:
        print('\nEpoch: %d' % epoch)
    train_losses = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_losses.update(loss.data.item(), inputs.shape[0])
        if verbose and (batch_idx+1) % disp_per_batch == 0:
            print(f"Batch [{batch_idx+1}/{len(train_loader)}]: "
                  f"Loss: {train_losses.val:.6f}")
    return train_losses.avg


def test(epoch, net, criterion, loader,
         verbose=False, disp_per_batch=10):
    net.eval()
    test_losses = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs).squeeze()
            loss = criterion(outputs, targets)
            test_losses.update(loss.data.item(), inputs.shape[0])
            if verbose and (batch_idx+1) % 10 == 0:
                print(f"Batch [{batch_idx+1}/{len(loader)}]: "
                      f"Loss: {test_losses.val:.6f}")
    return test_losses.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="l2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="meps_19")
    parser.add_argument("--data_path", type=str, default="./cqr/datasets/")
    parser.add_argument("--no_log_transform", action="store_true")
    parser.add_argument("--std_eps", type=float, default=1e-10)
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--cal_ratio", type=float, default=0.1)
    parser.add_argument("--recal_ratio", type=float, default=0.0)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cal_batch_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--network_mode", type=str, default="random_features")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="fixed")
    parser.add_argument("--decay_max_times", type=int, default=2)
    parser.add_argument("--decay_factor", type=float, default=0.1)
    parser.add_argument("--decay_patience", type=int, default=10)
    parser.add_argument("--decay_per_epoch", type=int, default=500)
    parser.add_argument("--disp_per_epoch", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--save_path", type=str, default='./runs/random_features')
    parser.add_argument("--wandb_project_name", type=str, default="random_features")
    parser.add_argument("--load_models", action="store_true")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--finetune_mode", type=str, default="conformal")
    parser.add_argument("--conformal_erm_epochs", type=int, default=1000)
    parser.add_argument("--conformal_erm_disp_per_epoch", type=int, default=50)
    parser.add_argument("--conformal_erm_lr", type=float, default=0.1)
    parser.add_argument("--conformal_erm_momentum", type=float, default=0.9)
    parser.add_argument("--conformal_erm_t_init", type=float, default=0.5)
    parser.add_argument("--conformal_erm_lam_init", type=float, default=1.0)
    parser.add_argument("--conformal_erm_lam_lr", type=float, default=0.1)
    parser.add_argument("--conformal_erm_lam_update", type=str, default="miscoverage")
    parser.add_argument("--conformal_erm_loss_cons", type=str, default="hinge")
    parser.add_argument("--cqr_only", action="store_true")
    parser.add_argument("--more_data", action="store_true")
    args = parser.parse_args()

    name = args.name + f"_seed={args.seed}"
    exp_path = os.path.join(args.save_path, args.dataset, name)
    model_path = os.path.join(exp_path, "models")
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    wandb_run = wandb.init(
        project=args.wandb_project_name,
        config=args,
        name=args.dataset + "/" + name,
        dir=exp_path,
    )

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
    print(torch.__version__)
    print(device)

    # Load the dataset
    X, y = datasets.GetDataset(args.dataset, args.data_path, log_transform=not args.no_log_transform)
    # Standardize x and y
    if not args.no_standardize:
        x_mean, x_std = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
        X = (X - x_mean) / (x_std + args.std_eps)
        y_mean, y_std = np.mean(y), np.std(y)
        y = (y - y_mean) / (y_std + args.std_eps)
        # import pdb; pdb.set_trace()

    # train test split
    n = X.shape[0]
    args.train_ratio = 1 - (args.cal_ratio + args.recal_ratio + args.test_ratio)
    n1, n2, n3 = int(args.train_ratio * n), int((args.train_ratio+args.cal_ratio) * n), int((1 - args.test_ratio) * n)
    perms = np.random.permutation(n)
    inds_train, inds_cal, inds_cal_all, inds_test = perms[:n1], perms[n1:n2], perms[n1:n3], perms[n3:]
    inds_traincal = perms[:n2]
    x_train, y_train = X[inds_train], y[inds_train]
    x_cal, y_cal = X[inds_cal], y[inds_cal]
    x_traincal, y_traincal = X[inds_traincal], y[inds_traincal]
    x_cal_all, y_cal_all = X[inds_cal_all], y[inds_cal_all]
    x_test, y_test = X[inds_test], y[inds_test]

    in_dim = x_train.shape[1]
    print("Dataset: %s" % args.dataset)
    print("Dimensions: train set (n=%d, d=%d), cal_all set (n=%d, d=%d), test set (n=%d, d=%d)" %
          (x_train.shape[0], x_train.shape[1], x_cal_all.shape[0], x_cal_all.shape[1], x_test.shape[0], x_test.shape[1]))

    # create dataloaders
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    cal_dataset = TensorDataset(torch.Tensor(x_cal), torch.Tensor(y_cal))
    traincal_dataset = TensorDataset(torch.Tensor(x_traincal), torch.Tensor(y_traincal))
    cal_all_dataset = TensorDataset(torch.Tensor(x_cal_all), torch.Tensor(y_cal_all))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    cal_loader = DataLoader(cal_dataset, batch_size=args.cal_batch_size, shuffle=True)
    traincal_loader = DataLoader(traincal_dataset, batch_size=args.batch_size, shuffle=True)
    cal_all_loader = DataLoader(cal_all_dataset, batch_size=args.cal_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.cal_batch_size, shuffle=False)

    if args.recal_ratio > 0:
        inds_recal = perms[n2:n3]
        x_recal, y_recal = X[inds_recal], y[inds_recal]
        recal_dataset = TensorDataset(torch.Tensor(x_recal), torch.Tensor(y_recal))
        recal_loader = DataLoader(recal_dataset, batch_size=args.cal_batch_size, shuffle=True)

    lambda1 = lambda ep: np.power(args.decay_factor, ep // args.decay_per_epoch)
    # Reset num epochs and decay
    if args.more_data:
        args.epochs = get_epochs(args.dataset)
        args.scheduler = "fixed"
        decay_epochs = [int(args.epochs * .9), int(args.epochs * .95)]
        lambda1 = lambda ep: lr_power_decay(ep, args.decay_factor, decay_epochs)


    # Create networks
    freeze_reps = (args.network_mode == "random_features")
    if args.mode in ["l2", "l1"]:
        # Train mean prediction network with L2 loss
        net = MLP(in_dim, out_dim=1, depth=args.depth, hidden_dim=args.width, freeze_reps=freeze_reps).to(device)
        if args.mode == "l2":
            criterion = nn.MSELoss(reduction="mean")
        elif args.mode == "l1":
            criterion = nn.L1Loss(reduction="mean")
    elif args.mode == "quantile":
        # Train quantile network with two-sided pinball loss
        net = MLP(in_dim, out_dim=2, depth=args.depth, hidden_dim=args.width,
                  freeze_reps=freeze_reps, dropout=args.dropout).to(device)
        criterion = TwoSidedPinballLoss(quantile_lo=args.alpha/2, quantile_hi=1-args.alpha/2, reduction="mean")

    # Train net, or optionally load existing models
    fn_template = "model.pt"
    load_path = args.load_path or model_path
    # try loading the existing model
    fn = os.path.join(load_path, fn_template)
    if args.load_models:
        if os.path.exists(fn):
            print(f"Loaded network")
            net.load_state_dict(torch.load(fn))
        else:
            raise FileNotFoundError
    else:
        print(f"Training network")
        wandb.define_metric(f"net/step")
        wandb.define_metric(f"net/*", step_metric=f"net/step")

        if args.optimizer == "sgd":
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        elif args.optimizer == "adam":
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
        if args.scheduler == "fixed":
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif args.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_factor,
                                                             patience=args.decay_patience)

        train_losses = list()
        val_losses = list()
        test_losses = list()
        curr_time = time.time()
        for epoch in range(args.epochs):
            # train for one epoch
            train_loss = train(epoch, net, optimizer, criterion, traincal_loader if args.more_data else train_loader)
            train_losses.append(train_loss)
            # val
            val_loss = test(epoch, net, criterion, cal_loader)
            val_losses.append(val_loss)
            # test
            test_loss = test(epoch, net, criterion, test_loader)
            test_losses.append(test_loss)
            wandb.log({
                f"net/lr": scheduler.optimizer.param_groups[0]['lr'],
                f"net/train_loss": train_loss,
                f"net/val_loss_cal": val_loss,
                f"net/test_loss": test_loss,
                f"net/step": epoch + 1
            })
            if (epoch + 1) % args.disp_per_epoch == 0:
                print(f"\nEpoch [{epoch + 1}]")
                print(f"Training loss = {train_loss:.4f}, Val loss = {val_loss:.4f}, Test loss = {test_loss:.4f}")
                print(f"Elapsed time = {time.time() - curr_time:.4f}s")
                curr_time = time.time()
            if args.scheduler == "fixed":
                scheduler.step()
            elif args.scheduler == "plateau":
                scheduler.step(val_loss)
                # if decayed for >= max_times + 1, then finish the optimization.
                if np.log(scheduler.optimizer.param_groups[0]['lr'] / args.lr) / np.log(args.decay_factor) > args.decay_max_times + 0.5:
                    break

        print(f"Training finished for network\n")
        torch.save(net.state_dict(), fn)

# Learn conformal/cqr interval using the recal dataset
# Note: alpha denotes miscoverage level (0.1)
critical_score = compute_critical_score(net, recal_loader, alpha=args.alpha,
                                        mode=args.mode, device=device)
coverage, length, corr, hsic = evaluate_coverage_length(
    net, test_loader, critical_score,
    mode=args.mode, device=device,
    return_corr=True
)
wandb.run.summary[f"net/critical_score"] = critical_score
wandb.run.summary[f"net/conformal_test_coverage"] = coverage
wandb.run.summary[f"net/conformal_test_length"] = length
wandb.run.summary[f"net/conformal_test_corr"] = corr
wandb.run.summary[f"net/conformal_test_hsic"] = hsic
print(f"Network + conformalize, test_coverage={coverage:.4f}, test_length={length:.4f}, test_corr={corr:.4f}, test_hsic={hsic:.4f}")


if args.cqr_only:
    sys.exit()

# Optimal length via smooth ERM
net_conformal = copy.deepcopy(net)
FreezeReps(net_conformal)

log_t = torch.tensor(np.log(args.conformal_erm_t_init)).to(device)
lam = torch.tensor(args.conformal_erm_lam_init).to(device)
log_t.requires_grad_()

optimizer = optim.SGD(list(net_conformal.parameters()) + [log_t],
                      lr=args.conformal_erm_lr, momentum=args.conformal_erm_momentum)

wandb.define_metric("conformal_erm/step")
wandb.define_metric("conformal_erm/*", step_metric="conformal_erm/step")

for epoch in range(args.conformal_erm_epochs):
    losses = AverageMeter()
    losses_obj = AverageMeter()
    losses_cons = AverageMeter()
    miscoverages = AverageMeter()
    for (i, batch) in enumerate(cal_loader):
        optimizer.zero_grad()
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = net_conformal(inputs).squeeze()
        t = torch.exp(log_t)

        if args.mode in ["l2", "l1"]:
            loss_obj = 2 * t
            cons_vals = t - torch.abs(outputs - targets)
        elif args.mode == "quantile":
            loss_obj = torch.mean(F.relu(outputs[:, 1] - outputs[:, 0] + 2 * t))
            cons_vals = torch.min(
                targets - outputs[:, 0] + t, t + outputs[:, 1] - targets
            )
        miscoverage = torch.mean((cons_vals < 0).float())

        # smoothified constraint loss
        if args.conformal_erm_loss_cons == "hinge":
            loss_cons = torch.mean(torch.max(-cons_vals + 1, torch.tensor(0.)))
        elif args.conformal_erm_loss_cons == "logistic":
            loss_cons = torch.mean(torch.log(1 + torch.exp(-cons_vals)))
        if args.finetune_mode == "conformal":
            loss = loss_obj + lam * loss_cons
        elif args.finetune_mode == "pinball":
            loss = criterion(outputs, targets)

        loss.backward()
        # gradient descent optimizer
        optimizer.step()
        # increase lambda by constraint violation
        if args.conformal_erm_lam_update == "miscoverage":
            cons_violation = torch.max(miscoverage - args.alpha, torch.tensor(0.))
        elif args.conformal_erm_lam_update == "loss":
            cons_violation = torch.max(loss_cons - args.alpha, torch.tensor(0.))
        with torch.no_grad():
            lam += args.conformal_erm_lam_lr * cons_violation

        losses.update(loss.data.item(), inputs.shape[0])
        losses_obj.update(loss_obj.data.item(), inputs.shape[0])
        losses_cons.update(loss_cons.data.item(), inputs.shape[0])
        miscoverages.update(miscoverage.data.item(), inputs.shape[0])

    log_dict = {
        "conformal_erm/loss": losses.avg,
        "conformal_erm/loss_obj": losses_obj.avg,
        "conformal_erm/loss_cons": losses_cons.avg,
        "conformal_erm/miscoverage": miscoverages.avg,
        "conformal_erm/lam": lam,
        "conformal_erm/step": epoch + 1,
    }
    wandb.log(log_dict)

    if (epoch + 1) % args.conformal_erm_disp_per_epoch == 0:
        print(f"Epoch [{epoch + 1}/{args.conformal_erm_epochs}]")
        print(f"Loss={losses.avg:.4f}")
        print(f"Loss_obj={losses_obj.avg:.4f}, Loss_cons={losses_cons.avg:.4f}, "
              f"Miscoverage_error={miscoverages.avg:.4f}, lam={lam:.4f}")


# Reconformalize and test the performances
# Step 1: conformalize on recalibration set
critical_score = compute_critical_score(net_conformal, recal_loader, alpha=args.alpha,
                                        mode=args.mode, device=device)
coverage, length, corr, hsic = evaluate_coverage_length(
    net_conformal, test_loader, critical_score,
    mode=args.mode, device=device,
    return_corr=True
)

# Step 2: evaluate on test set
print(f"recalibrated critical_score={critical_score:.4f}")
print(f"test_coverage={coverage:.4f}, test_length={length:.4f}, test_corr={corr:.4f}, test_hsic={hsic:.4f}")

wandb.run.summary["conformal_erm/recal_test_coverage"] = coverage
wandb.run.summary["conformal_erm/recal_test_length"] = length
wandb.run.summary["conformal_erm/recal_test_corr"] = corr
wandb.run.summary["conformal_erm/recal_test_hsic"] = hsic
wandb.run.summary["conformal_erm/recal_train_loss"] = test(epoch, net_conformal, criterion, train_loader)
wandb.run.summary["conformal_erm/recal_val_loss_cal"] = test(epoch, net_conformal, criterion, cal_loader)
wandb.run.summary["conformal_erm/recal_test_loss"] = test(epoch, net_conformal, criterion, test_loader)