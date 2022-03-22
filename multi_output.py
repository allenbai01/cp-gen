import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os, sys
import numpy as np
import time
import argparse
np.warnings.filterwarnings('ignore')

from models import MLP
from data_utils import get_rl_dataset
from utils import AverageMeter
import wandb

def train(epoch, net, optimizer, criterion,
          verbose=False, disp_per_batch=10):
    if verbose:
        print('\nEpoch: %d' % epoch)
    train_losses = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
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
    test_losses = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_losses.update(loss.data.item(), inputs.shape[0])
            if verbose and (batch_idx+1) % 10 == 0:
                print(f"Batch [{batch_idx+1}/{len(loader)}]: "
                      f"Loss: {test_losses.val:.6f}")
    return test_losses.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="cartpole",
                        help="{cartpole|half-cheetah|humanoid|swimmer|walker|ant|hopper}")
    parser.add_argument("--data_path", type=str, default="./rl_data/")
    parser.add_argument("--no_standardize", action="store_true")
    parser.add_argument("--cal_ratio", type=float, default=0.1)
    parser.add_argument("--recal_ratio", type=float, default=0.0)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--cal_batch_size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--decay_factor", type=float, default=0.1)
    parser.add_argument("--decay_per_epoch", type=int, default=500)
    parser.add_argument("--disp_per_epoch", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--save_path", type=str, default='./runs/multi_output')
    parser.add_argument("--wandb_project_name", type=str, default="multi_output")
    parser.add_argument("--load_models", action="store_true")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--conformal_erm_epochs", type=int, default=1000)
    parser.add_argument("--conformal_erm_disp_per_epoch", type=int, default=50)
    parser.add_argument("--conformal_erm_lr", type=float, default=0.01)
    parser.add_argument("--conformal_erm_lam_init", type=float, default=1.0)
    parser.add_argument("--conformal_erm_lam_lr", type=float, default=0.1)
    parser.add_argument("--conformal_erm_lam_update", type=str, default="miscoverage")
    parser.add_argument("--conformal_erm_lam_volume", type=float, default=1e4)
    parser.add_argument("--conformal_erm_loss_cons", type=str, default="hinge")
    parser.add_argument("--conformal_erm_recal_dataset", type=str, default="recal")
    parser.add_argument("--max_score_only", action="store_true")
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
    X, y = get_rl_dataset(args.data_path, args.dataset)

    # Standardize x and y
    if not args.no_standardize:
        x_mean, x_std = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
        X = (X - x_mean) / x_std
        y_mean, y_std = np.mean(y, axis=0, keepdims=True), np.std(y, axis=0, keepdims=True)
        y = (y - y_mean) / y_std

    # train test split
    n = X.shape[0]
    args.train_ratio = 1 - (args.cal_ratio + args.recal_ratio + args.test_ratio)
    n1, n2, n3 = int(args.train_ratio * n), int((args.train_ratio+args.cal_ratio) * n), int((1 - args.test_ratio) * n)
    perms = np.random.permutation(n)
    inds_train, inds_cal, inds_cal_all, inds_test = perms[:n1], perms[n1:n2], perms[n1:n3], perms[n3:]
    x_train, y_train = X[inds_train], y[inds_train]
    x_cal, y_cal = X[inds_cal], y[inds_cal]
    x_cal_all, y_cal_all = X[inds_cal_all], y[inds_cal_all]
    x_test, y_test = X[inds_test], y[inds_test]

    in_dim, out_dim = x_train.shape[1], y_train.shape[1]
    print("Dataset: %s" % args.dataset)
    print(
        "Dimensions: train set (n=%d, d=%d, d_out=%d), cal_all set (n=%d, d=%d, d_out=%d), test set (n=%d, d=%d, d_out=%d)" %
        (x_train.shape[0], x_train.shape[1], y_train.shape[1],
         x_cal_all.shape[0], x_cal_all.shape[1], y_cal_all.shape[1],
         x_test.shape[0], x_test.shape[1], y_test.shape[1])
    )

    # create dataloaders
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    cal_dataset = TensorDataset(torch.Tensor(x_cal), torch.Tensor(y_cal))
    cal_all_dataset = TensorDataset(torch.Tensor(x_cal_all), torch.Tensor(y_cal_all))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    cal_loader = DataLoader(cal_dataset, batch_size=args.cal_batch_size, shuffle=True)
    cal_all_loader = DataLoader(cal_all_dataset, batch_size=args.cal_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.cal_batch_size, shuffle=False)

    if args.recal_ratio > 0:
        inds_recal = perms[n2:n3]
        x_recal, y_recal = X[inds_recal], y[inds_recal]
        recal_dataset = TensorDataset(torch.Tensor(x_recal), torch.Tensor(y_recal))
        recal_loader = DataLoader(recal_dataset, batch_size=args.cal_batch_size, shuffle=True)

    # Create network
    net = MLP(in_dim, out_dim=out_dim, depth=args.depth, hidden_dim=args.width, freeze_reps=False).to(device)

    # Train model, or optionally load existing model
    fn_template = "model.pt"
    load_path = args.load_path or model_path
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

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        lambda1 = lambda ep: np.power(args.decay_factor, ep // args.decay_per_epoch)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        criterion = nn.MSELoss(reduction="mean")

        train_losses = list()
        test_losses = list()
        curr_time = time.time()
        for epoch in range(args.epochs):
            # train for one epoch
            train_loss = train(epoch, net, optimizer, criterion)
            train_losses.append(train_loss)
            # test
            test_loss = test(epoch, net, criterion, test_loader)
            test_losses.append(test_loss)
            wandb.log({f"net/train_loss": train_loss,
                       f"net/test_loss": test_loss,
                       f"net/step": epoch + 1})
            if (epoch + 1) % args.disp_per_epoch == 0:
                print(f"\nEpoch [{epoch + 1}]")
                print(f"Training loss = {train_loss:.4f}, Test loss = {test_loss:.4f}")
                print(f"Elapsed time = {time.time() - curr_time:.4f}s")
                curr_time = time.time()
            scheduler.step()

        print(f"Training finished\n")
        torch.save(net.state_dict(), fn)

# Method 0: Learn conformal interval [f_i(x) - t, f_j(x) + t] for same t
count = 0
scores = torch.zeros(len(cal_all_dataset)).to(device)
for (i, batch) in enumerate(cal_all_loader):
    inputs, targets = batch[0].to(device), batch[1].to(device)
    scores[count:(count + inputs.shape[0])] = torch.max(torch.abs(net(inputs) - targets), dim=1)[0]
    count += inputs.shape[0]
critical_score = torch.sort(scores)[0][int(np.ceil((1 - args.alpha) * (len(cal_all_dataset) + 1)))]

# test coverage and volume
coverage = 0.
volume = (critical_score ** out_dim)
for (i, batch) in enumerate(test_loader):
    inputs, targets = batch[0].to(device), batch[1].to(device)
    coverage += torch.sum(torch.prod(torch.abs(net(inputs) - targets) <= critical_score, dim=1))
coverage /= len(test_dataset)
print(f"Conformal with max score, test_coverage={coverage:.4f}, test_volume={volume:.4e}")
print(critical_score)
wandb.run.summary[f"max_score/test_coverage"] = coverage
wandb.run.summary[f"max_score/test_volume"] = volume

if args.max_score_only:
    sys.exit()


# Method 1: Learn Conformal interval [f_j(x) - t_j, f_j(x) + t_j] via union bound
# using the cal_all dataset
# Note: alpha denotes miscoverage level (0.1)
alpha_per_j = args.alpha / out_dim
critical_scores = torch.zeros([1, out_dim]).to(device)
for j in range(out_dim):
    scores = torch.zeros(len(cal_dataset)).to(device)
    count = 0
    for (i, batch) in enumerate(cal_loader):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        scores[count:(count + inputs.shape[0])] = torch.abs(net(inputs)[:, j] - targets[:, j])
        count += inputs.shape[0]
    critical_scores[0, j] = torch.sort(scores)[0][int(np.ceil((1-alpha_per_j) * (len(cal_dataset) + 1)))]

# test coverage and volume
coverage = 0.
volume = torch.prod(critical_scores)
for (i, batch) in enumerate(test_loader):
    inputs, targets = batch[0].to(device), batch[1].to(device)
    coverage += torch.sum(torch.prod(torch.abs(net(inputs) - targets) <= critical_scores, dim=1))
coverage /= len(test_dataset)
print(f"Conformal + union bound, test_coverage={coverage:.4f}, test_volume={volume:.4e}")
print(critical_scores)
wandb.run.summary[f"union/test_coverage"] = coverage
wandb.run.summary[f"union/test_volume"] = volume

# Method 2: Re-conformalize above interval
ts = torch.zeros(len(recal_dataset)).to(device)
count = 0
for (i, batch) in enumerate(recal_loader):
    inputs, targets = batch[0].to(device), batch[1].to(device)
    ts[count:(count + inputs.shape[0])] = torch.max(torch.abs(net(inputs) - targets) / critical_scores, dim=1)[0]
    count += inputs.shape[0]

critical_t = torch.sort(ts)[0][int(np.ceil((1-args.alpha) * (len(recal_dataset) + 1)))]
print(f"Critical_t={critical_t:.4f}")

# test coverage and volume
coverage = 0.
volume = torch.prod(critical_scores * critical_t)
for (i, batch) in enumerate(test_loader):
    inputs, targets = batch[0].to(device), batch[1].to(device)
    coverage += torch.sum(torch.prod(torch.abs(net(inputs) - targets) <= critical_scores * critical_t, dim=1))
coverage /= len(test_dataset)
print(f"Conformal + union bound + recalib, test_coverage={coverage:.4f}, test_volume={volume:.4e}")
print(critical_scores * critical_t)
wandb.run.summary[f"union_recal/test_coverage"] = coverage
wandb.run.summary[f"union_recal/test_volume"] = volume
wandb.run.summary[f"union_recal/t_recal"] = critical_t

# Method 3: Optimal volume via smooth ERM
logit_ts = -5.0 * torch.ones([1, out_dim]).to(device)
lam_volume = args.conformal_erm_lam_volume
lam = torch.tensor(args.conformal_erm_lam_init).to(device)
logit_ts.requires_grad = True

optimizer = optim.SGD([logit_ts], lr=args.conformal_erm_lr, momentum=0.)

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
        outputs = net(inputs)
        ts = torch.exp(logit_ts)
        loss_obj = torch.prod(ts)
        cons_vals = 1.0 - torch.max(torch.abs(targets - outputs) / ts, dim=1)[0]
        miscoverage = torch.mean((cons_vals < 0).float())
        # smoothified constraint loss
        if args.conformal_erm_loss_cons == "hinge":
            loss_cons = torch.mean(torch.max(-cons_vals + 1, torch.tensor(0.)))
        elif args.conformal_erm_loss_cons == "logistic":
            loss_cons = torch.mean(torch.log(1 + torch.exp(-cons_vals)))

        loss = lam_volume * loss_obj + lam * loss_cons
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
        print(f"ts={ts}")


# Reconformalize and test the performances
critical_scores = torch.exp(logit_ts).clone()
print(f"Initial critical_scores={critical_scores}")

count = 0
if args.conformal_erm_recal_dataset == "recal":
    cerm_recal_set, cerm_recal_loader = recal_dataset, recal_loader
elif args.conformal_erm_recal_dataset == "cal":
    cerm_recal_set, cerm_recal_loader = cal_dataset, cal_loader

new_ts = torch.zeros(len(cerm_recal_set)).to(device)
for (i, batch) in enumerate(cerm_recal_loader):
    inputs, targets = batch[0].to(device), batch[1].to(device)
    # import pdb; pdb.set_trace()
    new_ts[count:(count + inputs.shape[0])] = torch.max(torch.abs(net(inputs) - targets) / critical_scores, dim=1)[0]
    count += inputs.shape[0]
critical_t = torch.sort(new_ts)[0][int(np.ceil((1-args.alpha) * (len(cerm_recal_set) + 1)))]
print(f"Critical_t={critical_t:.4f}")

# test coverage and volume
coverage = 0.
volume = torch.prod(critical_scores * critical_t)
for (i, batch) in enumerate(test_loader):
    inputs, targets = batch[0].to(device), batch[1].to(device)
    coverage += torch.sum(torch.prod(torch.abs(net(inputs) - targets) <= critical_scores * critical_t, dim=1))
coverage /= len(test_dataset)
print(f"Conformal ERM method, test_coverage={coverage:.4f}, test_volume={volume:.4e}")
print(critical_scores * critical_t)
wandb.run.summary[f"conformal_erm/recal_test_coverage"] = coverage
wandb.run.summary[f"conformal_erm/recal_test_volume"] = volume
wandb.run.summary[f"conformal_erm/t_recal"] = critical_t
