"""Experiment 005: How wrong is a model's confidence?

Registered plan: ../plan.md (frozen, approved, 2026-09-08).

Steps:
1. Load edadaltocg/resnet34_cifar100 (public, MIT) - see model.py.
2. Compute logits for the whole official CIFAR-100 test set (10,000 images).
3. Split the test set into two equal halves (fixed seed, for
   reproducibility): `dev` (fits the temperature T) and `eval` (final
   evaluation only).
4. Compute ECE (15 bins) and top-1 accuracy on `eval`, before and after
   T-scaling.
5. Write the raw numbers to results.json.

Fetch the test data first:
  curl -sL -A "Mozilla/5.0" \
    "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet" \
    -o data/cifar100_test.parquet
"""

import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from model import CIFAR100_MEAN, CIFAR100_STD, build_resnet34_cifar100

SEED = 1
N_BINS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = Path(__file__).parent


def expected_calibration_error(confidences: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor, n_bins: int = N_BINS) -> float:
    correct = predictions.eq(labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    n = confidences.size(0)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        bin_size = in_bin.sum().item()
        if bin_size == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_size / n) * (bin_acc - bin_conf).abs()
    return ece.item()


def reliability_bins(confidences: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor, n_bins: int = N_BINS):
    correct = predictions.eq(labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bins = []
    n = confidences.size(0)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        bin_size = in_bin.sum().item()
        bins.append({
            "range": [round(lo.item(), 3), round(hi.item(), 3)],
            "count": bin_size,
            "fraction_of_data": round(bin_size / n, 4),
            "avg_confidence": round(confidences[in_bin].mean().item(), 4) if bin_size else None,
            "avg_accuracy": round(correct[in_bin].mean().item(), 4) if bin_size else None,
        })
    return bins


class Cifar100ParquetTest(Dataset):
    """The official CIFAR-100 test split (10,000 images), taken from the
    uoft-cs/cifar100 HF dataset as parquet instead of torchvision's own
    mirror, because the latter was extremely slow from this network."""

    def __init__(self, parquet_path: str, transform):
        table = pq.read_table(parquet_path, columns=["img", "fine_label"])
        self.images = table.column("img").to_pylist()
        self.labels = table.column("fine_label").to_pylist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_bytes = self.images[idx]["bytes"]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self.transform(image), self.labels[idx]


@torch.no_grad()
def collect_logits(model, loader) -> tuple[torch.Tensor, torch.Tensor]:
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def evaluate(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> dict:
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracy = predictions.eq(labels).float().mean().item()
    ece = expected_calibration_error(confidences, predictions, labels)
    bins = reliability_bins(confidences, predictions, labels)
    return {
        "temperature": temperature,
        "accuracy": accuracy,
        "ece": ece,
        "reliability_bins": bins,
    }


def main():
    torch.manual_seed(SEED)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_set = Cifar100ParquetTest(str(HERE / "data" / "cifar100_test.parquet"), transform=transform)

    n = len(test_set)
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(n, generator=generator).tolist()
    dev_indices = permutation[: n // 2]
    eval_indices = permutation[n // 2:]

    dev_set = Subset(test_set, dev_indices)
    eval_set = Subset(test_set, eval_indices)

    dev_loader = DataLoader(dev_set, batch_size=256, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_set, batch_size=256, shuffle=False, num_workers=0)

    model = build_resnet34_cifar100(str(HERE / "resnet34_cifar100.bin")).to(DEVICE)

    print(f"dev size: {len(dev_set)}, eval size: {len(eval_set)}, device: {DEVICE}")

    dev_logits, dev_labels = collect_logits(model, dev_loader)
    eval_logits, eval_labels = collect_logits(model, eval_loader)

    before = evaluate(eval_logits, eval_labels, temperature=1.0)
    print(f"BEFORE scaling - eval accuracy: {before['accuracy']:.4f}, ECE: {before['ece']:.4f}")

    temperature = fit_temperature(dev_logits, dev_labels)
    print(f"fitted temperature (on dev): {temperature:.4f}")

    after = evaluate(eval_logits, eval_labels, temperature=temperature)
    print(f"AFTER scaling  - eval accuracy: {after['accuracy']:.4f}, ECE: {after['ece']:.4f}")

    relative_ece_reduction = (before["ece"] - after["ece"]) / before["ece"] if before["ece"] > 0 else 0.0

    results = {
        "seed": SEED,
        "n_bins": N_BINS,
        "dev_size": len(dev_set),
        "eval_size": len(eval_set),
        "fitted_temperature": temperature,
        "before": before,
        "after": after,
        "relative_ece_reduction": relative_ece_reduction,
        "kill_gates": {
            "KG1_ece_before_lt_3pct": before["ece"] < 0.03,
            "KG2_relative_reduction_lt_30pct": relative_ece_reduction < 0.30,
            "KG3_accuracy_unchanged": abs(before["accuracy"] - after["accuracy"]) < 1e-6,
        },
    }

    out_path = HERE.parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
