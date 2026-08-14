# Catastrophic forgetting on Permuted MNIST, plain sequential fine-tuning
# vs EWC (Kirkpatrick et al. 2017, arXiv 1612.00796). Small MLP (400-400),
# same scale as the original paper. Public MNIST data only.
import io
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS_PER_TASK = 5
BATCH_SIZE = 128
LR = 1e-3
EWC_LAMBDA = 1000.0
FISHER_SAMPLES = 1000

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_mnist():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    root = os.path.join(HERE, "data")
    train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    return train, test


def to_tensors(ds):
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.tensor([ds[i][1] for i in range(len(ds))])
    return xs, ys


def permute(x, perm):
    return x[:, perm]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_epochs(model, x, y, epochs, ewc=None):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            if ewc is not None:
                loss = loss + ewc.penalty(model)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f"    epoch {ep+1}/{epochs} loss={total_loss/len(ds):.4f}", flush=True)


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    x, y = x.to(DEVICE), y.to(DEVICE)
    out = model(x)
    pred = out.argmax(dim=1)
    return (pred == y).float().mean().item()


class EWC:
    """Diagonal Fisher approximation computed after Task A, penalising drift
    away from the Task-A-optimal parameters."""

    def __init__(self, model, x_a, y_a, n_samples=FISHER_SAMPLES):
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        idx = torch.randperm(x_a.size(0))[:n_samples]
        xs, ys = x_a[idx].to(DEVICE), y_a[idx].to(DEVICE)
        for i in range(xs.size(0)):
            model.zero_grad()
            out = model(xs[i:i+1])
            loss = F.cross_entropy(out, ys[i:i+1])
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2 / n_samples
        model.zero_grad()

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            loss = loss + (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return EWC_LAMBDA / 2 * loss


def main():
    t0 = time.time()
    print("device:", DEVICE, flush=True)
    train_ds, test_ds = load_mnist()
    print("loading tensors...", flush=True)
    x_train, y_train = to_tensors(train_ds)
    x_test, y_test = to_tensors(test_ds)

    rng = np.random.RandomState(SEED)
    perm = torch.tensor(rng.permutation(784))

    xA_train, yA_train = x_train, y_train
    xA_test, yA_test = x_test, y_test
    xB_train, yB_train = permute(x_train, perm), y_train
    xB_test, yB_test = permute(x_test, perm), y_test

    results = {}

    # --- Baseline: Task A only ---
    print("\n=== Train Task A ===", flush=True)
    model_a = MLP().to(DEVICE)
    train_epochs(model_a, xA_train, yA_train, EPOCHS_PER_TASK)
    acc_a_after_a = evaluate(model_a, xA_test, yA_test)
    print(f"  Task A acc after Task A: {acc_a_after_a:.4f}", flush=True)
    results["acc_A_after_A"] = acc_a_after_a
    state_after_a = {k: v.clone() for k, v in model_a.state_dict().items()}

    # --- Plain sequential fine-tuning: Task A -> Task B, no protection ---
    print("\n=== Continue to Task B (plain, no EWC) ===", flush=True)
    model_plain = MLP().to(DEVICE)
    model_plain.load_state_dict(state_after_a)
    train_epochs(model_plain, xB_train, yB_train, EPOCHS_PER_TASK)
    acc_a_after_b_plain = evaluate(model_plain, xA_test, yA_test)
    acc_b_after_b_plain = evaluate(model_plain, xB_test, yB_test)
    print(f"  Task A acc after plain Task B: {acc_a_after_b_plain:.4f}", flush=True)
    print(f"  Task B acc (plain): {acc_b_after_b_plain:.4f}", flush=True)
    results["acc_A_after_B_plain"] = acc_a_after_b_plain
    results["acc_B_after_B_plain"] = acc_b_after_b_plain

    # --- EWC-protected sequential fine-tuning: Task A -> Task B ---
    print("\n=== Continue to Task B (EWC) ===", flush=True)
    model_ewc = MLP().to(DEVICE)
    model_ewc.load_state_dict(state_after_a)
    print("  computing Fisher information from Task A...", flush=True)
    ewc = EWC(model_ewc, xA_train, yA_train)
    train_epochs(model_ewc, xB_train, yB_train, EPOCHS_PER_TASK, ewc=ewc)
    acc_a_after_b_ewc = evaluate(model_ewc, xA_test, yA_test)
    acc_b_after_b_ewc = evaluate(model_ewc, xB_test, yB_test)
    print(f"  Task A acc after EWC Task B: {acc_a_after_b_ewc:.4f}", flush=True)
    print(f"  Task B acc (EWC): {acc_b_after_b_ewc:.4f}", flush=True)
    results["acc_A_after_B_ewc"] = acc_a_after_b_ewc
    results["acc_B_after_B_ewc"] = acc_b_after_b_ewc

    # --- Kill-gate verdicts ---
    drop_plain = results["acc_A_after_A"] - results["acc_A_after_B_plain"]
    recovered = results["acc_A_after_B_ewc"] - results["acc_A_after_B_plain"]
    recovery_frac = recovered / drop_plain if drop_plain > 0 else None

    kg1_pass = drop_plain >= 0.20
    kg2_pass = recovery_frac is not None and recovery_frac >= 0.50

    results["drop_plain_pp"] = drop_plain * 100
    results["recovery_fraction"] = recovery_frac
    results["KG1_forgetting_demonstrated"] = "PASSED" if kg1_pass else "FIRED (H1 rejected)"
    results["KG2_ewc_mitigates"] = "PASSED" if kg2_pass else "FIRED (H2 rejected)"
    results["config"] = {
        "epochs_per_task": EPOCHS_PER_TASK, "batch_size": BATCH_SIZE, "lr": LR,
        "ewc_lambda": EWC_LAMBDA, "fisher_samples": FISHER_SAMPLES, "seed": SEED,
        "device": str(DEVICE), "wall_clock_seconds": time.time() - t0,
    }

    json.dump(results, open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8"), indent=2)

    print("\n=== SUMMARY ===")
    print(f"Task A acc after Task A:        {results['acc_A_after_A']:.4f}")
    print(f"Task A acc after plain Task B:  {results['acc_A_after_B_plain']:.4f}  (drop {drop_plain*100:.1f} pp)")
    print(f"Task A acc after EWC Task B:    {results['acc_A_after_B_ewc']:.4f}  (recovered {recovered*100:.1f} pp = {recovery_frac*100:.1f}% of the drop)" if recovery_frac is not None else "")
    print(f"\nKG1 (drop >= 20pp): {results['KG1_forgetting_demonstrated']}")
    print(f"KG2 (recovery >= 50%): {results['KG2_ewc_mitigates']}")
    print(f"\nwall clock: {results['config']['wall_clock_seconds']:.1f}s")


if __name__ == "__main__":
    main()
