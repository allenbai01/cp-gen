import os

from utils import *
from multi_conformal import LACPredictionSets

# Import other standard packages
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="ImageNet",
                    help="{ImageNet|ImageNetV2}")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--name", type=str, default="")
parser.add_argument("--save_path", type=str, default='./runs/imagenet')
parser.add_argument("--load_models", action="store_true")
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--conformal_erm_p", type=float, default=1.0)
parser.add_argument("--conformal_erm_epochs", type=int, default=500)
parser.add_argument("--conformal_erm_disp_per_epoch", type=int, default=10)
parser.add_argument("--conformal_erm_compute_size_per_epoch", type=int, default=10)
parser.add_argument("--conformal_erm_lr", type=float, default=0.01)
parser.add_argument("--conformal_erm_t_init", type=float, default=0.5)
parser.add_argument("--conformal_erm_lam_init", type=float, default=1.0)
parser.add_argument("--conformal_erm_lam_lr", type=float, default=0.01)
parser.add_argument("--conformal_erm_lam_update", type=str, default="loss")
parser.add_argument("--conformal_erm_loss_cons", type=str, default="hinge")
args = parser.parse_args()

name = args.name + f"_seed={args.seed}"
exp_path = os.path.join(args.save_path, args.dataset, name)
model_path = os.path.join(exp_path, "models")
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

wandb_run = wandb.init(
    project="imagenet",
    config=args,
    name=args.dataset + "/" + name,
    dir=exp_path,
)

# Fix the random seed for reproducibility (you can change this, of course)
np.random.seed(seed=args.seed)
torch.manual_seed(args.seed)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)
print(torch.__version__)
print(device)


# Load logits
modelnames = [
    'ResNeXt101', 'ResNet152', 'ResNet101', 'DenseNet161',
    'ResNet18', 'ResNet50', 'VGG16', 'Inception', 'ShuffleNet',
]
n_models = len(modelnames)
if args.dataset == "ImageNet":
    n_cal, n_recal = 10000, 10000
elif args.dataset == "ImageNetV2":
    n_cal, n_recal = 4000, 1000

logit_datasets = list()
for model in modelnames:
    l = my_get_logits_dataset(model, args.dataset)
    logit_datasets.append(l)
    print(f"Logits from model {model} loaded.")

# Create datasets and loaders
combined_tensors = [l.tensors[0] for l in logit_datasets] + [logit_datasets[0].tensors[1]]
combined_dataset = torch.utils.data.TensorDataset(*combined_tensors)
n_test = len(combined_dataset) - n_cal - n_recal
combined_dataset_cal, combined_dataset_recal, combined_dataset_test = torch.utils.data.random_split(
    combined_dataset, [n_cal, n_recal, n_test]
)

cal_loader = torch.utils.data.DataLoader(
    combined_dataset_cal, batch_size=args.batch_size, shuffle=True,
    pin_memory=True,
)
recal_loader = torch.utils.data.DataLoader(
    combined_dataset_recal, batch_size=args.batch_size, shuffle=True,
    pin_memory=True,
)
test_loader = torch.utils.data.DataLoader(
    combined_dataset_test, batch_size=args.batch_size, shuffle=False,
    pin_memory=True,
)

# Optimization based ensemble via ERM
log_weights = torch.zeros([n_models]).to(device)
log_t = torch.tensor([np.log(args.conformal_erm_t_init)]).to(device)
log_weights.requires_grad = True
log_t.requires_grad = True

lam = torch.tensor(1.).to(device)
# eta = 0.01
# eta_lam = 0.01
optimizer = optim.SGD([log_weights, log_t], lr=args.conformal_erm_lr, momentum=0.)

best_weights, best_size = None, 1000
cal_tensors = combined_dataset[combined_dataset_cal.indices]

wandb.define_metric("conformal_erm/step")
wandb.define_metric("conformal_erm/*", step_metric="conformal_erm/step")

for epoch in range(args.conformal_erm_epochs):
    losses = AverageMeter()
    losses_obj = AverageMeter()
    losses_cons = AverageMeter()
    sizes = AverageMeter()
    miscoverages = AverageMeter()
    for batch in cal_loader:
        optimizer.zero_grad()
        logits, targets = batch[:-1], batch[-1]
        # set_trace()
        mixed_probs = torch.zeros_like(logits[0]).to(device)
        weights, t = F.softmax(log_weights), torch.exp(log_t)
        for (i, logit) in enumerate(logits):
            mixed_probs += weights[i] * F.softmax(logit.to(device), dim=1)
        size = (mixed_probs > t).float().sum(dim=1).mean()
        loss_obj = ((F.relu(mixed_probs - t)) ** args.conformal_erm_p).sum(dim=1).mean()
        cons_vals = mixed_probs[range(mixed_probs.shape[0]), targets] - t
        miscoverage = torch.mean((cons_vals < 0).float())

        # smoothified constraint loss
        if args.conformal_erm_loss_cons == "hinge":
            loss_cons = torch.mean(torch.max(-cons_vals + 1, torch.tensor(0.)))
        elif args.conformal_erm_loss_cons == "logistic":
            loss_cons = torch.mean(torch.log(1 + torch.exp(-cons_vals)))
        loss = loss_obj + lam * loss_cons

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

        losses.update(loss.data.item(), targets.shape[0])
        losses_obj.update(loss_obj.data.item(), targets.shape[0])
        losses_cons.update(loss_cons.data.item(), targets.shape[0])
        sizes.update(size.data.item(), targets.shape[0])
        miscoverages.update(miscoverage.data.item(), targets.shape[0])

    log_dict = {
        "conformal_erm/loss": losses.avg,
        "conformal_erm/loss_obj": losses_obj.avg,
        "conformal_erm/loss_cons": losses_cons.avg,
        "conformal_erm/size": sizes.avg,
        "conformal_erm/miscoverage": miscoverages.avg,
        "conformal_erm/lam": lam,
        "conformal_erm/step": epoch + 1,
    }
    wandb.log(log_dict)

    if (epoch + 1) % args.conformal_erm_disp_per_epoch == 0:
        print(f"Epoch [{epoch + 1}/{args.conformal_erm_epochs}]")
        print(f"weights={weights.detach().cpu()}:")
        print(f"size={sizes.avg:.4f}")
        print(f"Loss={losses.avg:.4f}")
        print(f"Loss_obj={losses_obj.avg:.4f}, Loss_cons={losses_cons.avg:.4f}, "
              f"Miscoverage_error={miscoverages.avg:.4f}, lam={lam:.4f}")

    if (epoch + 1) % args.conformal_erm_compute_size_per_epoch == 0:
        wts = weights.detach().cpu()
        cal_mixed_probs = torch.zeros_like(cal_tensors[0])
        weights = weights.cpu()
        for i in range(n_models):
            cal_mixed_probs += wts[i] * F.softmax(cal_tensors[i], dim=1)
        cal_mixed_logits = torch.log(cal_mixed_probs).detach()
        cal_mixed_dataset = torch.utils.data.TensorDataset(cal_mixed_logits, cal_tensors[-1])
        top1_avg, top5_avg, cvg_avg, sz_avg = LACPredictionSets(
            cal_mixed_dataset, cal_mixed_dataset, args.alpha, args.batch_size,
            temp_scale=False
        )
        wandb.log({
            "conformal_erm/size_cal": sz_avg,
            "conformal_erm/cvg_cal": cvg_avg,
        })
        if sz_avg < best_size:
            best_size = sz_avg
            best_weights = wts
            print(f"New best weights {wts}")
            print(f"Avg_size_cal={sz_avg:.4f}")


# Test the resulting weights
weights_list, names_list = list(), list()
eye = torch.eye(n_models)
for i in range(n_models):
    weights_list.append(eye[i, :])
    names_list.append(modelnames[i])
unif_weights = torch.ones([n_models]) / n_models
weights_list += [unif_weights, weights.cpu(), best_weights]
names_list += ["uniform", "optim", "optim_best"]

for (name, wts) in zip(names_list, weights_list):
    if "optim" in name:
        recal_tensors = combined_dataset[combined_dataset_recal.indices]
    else:
        recal_tensors = combined_dataset[combined_dataset_cal.indices + combined_dataset_recal.indices]
    test_tensors = combined_dataset[combined_dataset_test.indices]
    recal_mixed_probs = torch.zeros_like(recal_tensors[0])
    test_mixed_probs = torch.zeros_like(test_tensors[0])
    weights = weights.cpu()
    for i in range(n_models):
        recal_mixed_probs += wts[i] * F.softmax(recal_tensors[i], dim=1)
        test_mixed_probs += wts[i] * F.softmax(test_tensors[i], dim=1)
    recal_mixed_logits = torch.log(recal_mixed_probs).detach()
    test_mixed_logits = torch.log(test_mixed_probs).detach()

    recal_mixed_dataset = torch.utils.data.TensorDataset(recal_mixed_logits, recal_tensors[-1])
    test_mixed_dataset = torch.utils.data.TensorDataset(test_mixed_logits, test_tensors[-1])

    # set_trace()

    top1_avg, top5_avg, cvg_avg, sz_avg = LACPredictionSets(
        recal_mixed_dataset, test_mixed_dataset, args.alpha, args.batch_size,
        temp_scale=False
    )
    print(f"weights={wts}:")
    print(f"top1_avg={top1_avg:.4f}, top5_avg={top5_avg:.4f}")
    print(f"coverage_test={cvg_avg:.4f}, size_test={sz_avg:.4f}")

    wandb.run.summary[f"{name}/recal_coverage"] = cvg_avg
    wandb.run.summary[f"{name}/recal_size"] = sz_avg