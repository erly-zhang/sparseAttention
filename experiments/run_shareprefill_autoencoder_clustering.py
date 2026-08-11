#!/usr/bin/env python3
"""Build fixed-K representative-head groups from attention-map autoencoding.

This follows SharePrefill's offline representation pipeline: full attention
score maps are encoded by a convolutional autoencoder into 64 dimensions,
normalized, and hierarchically clustered. The paper chooses clusters with a
distance threshold. This experiment instead cuts each layer into a requested
fixed K and chooses the latent-space medoid required by the project runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import cut_tree, linkage
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.run_jsd_grouped_efficiency_experiment import (  # noqa: E402
    split_canonical_samples,
)
from experiments.run_shared_layer_mask_experiment import (  # noqa: E402
    build_prompt,
    load_longbench_v2_samples,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class SharePrefillAutoencoder(nn.Module):
    """Paper Table 3 architecture for a 1936 x 1936 attention map."""

    def __init__(self, map_size: int = 1936, latent_dim: int = 64) -> None:
        super().__init__()
        if map_size % 16:
            raise ValueError("map_size must be divisible by 16")
        self.map_size = int(map_size)
        self.latent_dim = int(latent_dim)
        pooled = map_size // 16
        flattened = 32 * pooled * pooled
        self.pooled_size = pooled
        self.flattened_size = flattened
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self.encoder_linear = nn.Linear(flattened, latent_dim)
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, flattened),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 8, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )

    def encode(self, maps: torch.Tensor) -> torch.Tensor:
        features = self.encoder_conv(maps)
        return self.encoder_linear(features.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.decoder_linear(latent).view(
            -1, 32, self.pooled_size, self.pooled_size
        )
        return self.decoder_conv(features)

    def forward(self, maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(maps)
        return self.decode(latent), latent


class MeanAttentionMapDataset(Dataset[torch.Tensor]):
    """Lazily read one mean attention map for each (layer, head)."""

    def __init__(
        self,
        mean_map_dir: Path,
        *,
        num_layers: int,
        num_heads: int,
    ) -> None:
        self.mean_map_dir = mean_map_dir
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

    def __len__(self) -> int:
        return self.num_layers * self.num_heads

    def __getitem__(self, index: int) -> torch.Tensor:
        layer = index // self.num_heads
        head = index % self.num_heads
        path = self.mean_map_dir / f"layer_{layer:02d}.npy"
        layer_maps = np.load(path, mmap_mode="r")
        attention = np.asarray(layer_maps[head], dtype=np.float32)
        return torch.from_numpy(attention.copy()).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SharePrefill attention-map autoencoder clustering"
    )
    parser.add_argument(
        "stage", choices=["collect", "train", "cluster", "all"]
    )
    parser.add_argument(
        "--model_name_or_path",
        default="/home/ubuntu/work/model/Qwen2.5-7B",
    )
    parser.add_argument(
        "--data_path",
        default=(
            "/home/ubuntu/work/experiments/data/"
            "longbench_v2_32k_full_7b.jsonl"
        ),
    )
    parser.add_argument(
        "--calibration_format",
        choices=["longbench", "prompt_jsonl"],
        default="longbench",
        help=(
            "longbench uses the canonical reserved rows; prompt_jsonl reads "
            "records with _id and prompt fields."
        ),
    )
    parser.add_argument(
        "--artifact_dir",
        default=(
            "/home/ubuntu/work/experiments/outputs/"
            "shareprefill_autoencoder_calibration"
        ),
    )
    parser.add_argument("--head_selection_num_samples", type=int, default=3)
    parser.add_argument("--map_size", type=int, default=1936)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-7)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--groups", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()
    if args.map_size != 1936:
        raise ValueError("Paper-compatible autoencoder requires map_size=1936")
    if args.latent_dim != 64:
        raise ValueError("Paper-compatible autoencoder requires latent_dim=64")
    return args


def _loader_args(args: argparse.Namespace) -> argparse.Namespace:
    args.num_samples = max(args.head_selection_num_samples, 3)
    args.samples_per_domain = None
    args.domains = None
    args.sub_domain = None
    args.difficulty = None
    args.length = None
    args.start_line = 0
    return args


def load_calibration_samples(
    args: argparse.Namespace,
) -> List[Mapping[str, Any]]:
    if args.calibration_format == "prompt_jsonl":
        samples: List[Mapping[str, Any]] = []
        with Path(args.data_path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if "_id" not in sample or "prompt" not in sample:
                    raise ValueError(
                        "prompt_jsonl rows require _id and prompt fields"
                    )
                samples.append(sample)
        if len(samples) != args.head_selection_num_samples:
            raise ValueError(
                f"Expected {args.head_selection_num_samples} calibration "
                f"rows, found {len(samples)}"
            )
        return samples
    samples = load_longbench_v2_samples(_loader_args(args))
    head_samples, _ = split_canonical_samples(
        samples,
        head_selection_num_samples=args.head_selection_num_samples,
        eval_num_samples=0,
    )
    if len(head_samples) != args.head_selection_num_samples:
        raise ValueError("Canonical head-selection sample count changed")
    return list(head_samples)


def build_calibration_prompt(sample: Mapping[str, Any]) -> str:
    if "prompt" in sample:
        return str(sample["prompt"])
    return build_prompt(dict(sample))


def _sample_fingerprint(
    sample_ids: Sequence[str], input_ids: Sequence[torch.Tensor]
) -> str:
    digest = hashlib.sha256()
    for sample_id, tokens in zip(sample_ids, input_ids):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            ",".join(str(int(token)) for token in tokens.tolist()).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


@torch.inference_mode()
def collect_attention_maps(args: argparse.Namespace) -> Dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    map_root = artifact_dir / "attention_maps"
    map_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    samples = load_calibration_samples(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    sample_ids: List[str] = []
    token_records: List[torch.Tensor] = []

    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("_id"))
        prompt = build_calibration_prompt(sample)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.map_size,
            padding=False,
        )
        input_ids = encoded["input_ids"][0]
        if input_ids.numel() != args.map_size:
            raise ValueError(
                f"Calibration sample {sample_id} has only "
                f"{input_ids.numel()} tokens; expected {args.map_size}"
            )
        sample_ids.append(sample_id)
        token_records.append(input_ids.clone())
        sample_dir = map_root / f"sample_{sample_index:02d}_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        handles = []

        def make_hook(layer_idx: int):
            def save_attention(_module, _inputs, output):
                if not isinstance(output, tuple) or len(output) < 2:
                    raise RuntimeError("Unexpected eager attention output")
                attention = output[1]
                if attention is None:
                    raise RuntimeError("Eager attention weights are missing")
                array = (
                    attention[0]
                    .detach()
                    .to(device="cpu", dtype=torch.float16)
                    .numpy()
                )
                if array.shape != (
                    num_heads,
                    args.map_size,
                    args.map_size,
                ):
                    raise ValueError(
                        f"Layer {layer_idx} map shape changed: {array.shape}"
                    )
                np.save(sample_dir / f"layer_{layer_idx:02d}.npy", array)
                replacement = list(output)
                replacement[1] = None
                return tuple(replacement)

            return save_attention

        for layer_idx, layer in enumerate(model.model.layers):
            handles.append(
                layer.self_attn.register_forward_hook(make_hook(layer_idx))
            )
        try:
            device_inputs = {
                key: value.to(args.device) for key, value in encoded.items()
            }
            model(
                **device_inputs,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        logger.info(
            "Collected full attention maps for calibration %d/%d: %s",
            sample_index + 1,
            len(samples),
            sample_id,
        )
        torch.cuda.empty_cache()

    mean_dir = artifact_dir / "mean_attention_maps"
    mean_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs = sorted(map_root.glob("sample_*"))
    for layer_idx in range(num_layers):
        mean_maps = np.zeros(
            (num_heads, args.map_size, args.map_size), dtype=np.float32
        )
        for sample_dir in sample_dirs:
            mean_maps += np.load(
                sample_dir / f"layer_{layer_idx:02d}.npy", mmap_mode="r"
            ).astype(np.float32)
        mean_maps /= len(sample_dirs)
        np.save(
            mean_dir / f"layer_{layer_idx:02d}.npy",
            mean_maps.astype(np.float16),
        )
        logger.info("Saved mean attention maps for layer %d", layer_idx)

    manifest = {
        "schema_version": 1,
        "source": (
            "canonical_head_selection_samples_only"
            if args.calibration_format == "longbench"
            else "benchmark_specific_prompt_jsonl"
        ),
        "calibration_format": args.calibration_format,
        "model_name_or_path": args.model_name_or_path,
        "data_path": args.data_path,
        "sample_ids": sample_ids,
        "num_samples": len(sample_ids),
        "map_size": args.map_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dtype": "float16",
        "input_truncation": "tokenizer_right_truncation_to_1936",
        "input_fingerprint_sha256": _sample_fingerprint(
            sample_ids, token_records
        ),
        "evaluation_sample_usage": False,
    }
    (artifact_dir / "attention_map_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    del model
    torch.cuda.empty_cache()
    return manifest


def _load_manifest(artifact_dir: Path) -> Dict[str, Any]:
    path = artifact_dir / "attention_map_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Collect attention maps first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def train_autoencoder(args: argparse.Namespace) -> Dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    manifest = _load_manifest(artifact_dir)
    dataset = MeanAttentionMapDataset(
        artifact_dir / "mean_attention_maps",
        num_layers=int(manifest["num_layers"]),
        num_heads=int(manifest["num_heads"]),
    )
    indices = list(range(len(dataset)))
    validation = [index for index in indices if index % 10 == 0]
    training = [index for index in indices if index % 10 != 0]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, training),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model = SharePrefillAutoencoder(
        map_size=args.map_size, latent_dim=args.latent_dim
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler_enabled = args.device.startswith("cuda")
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    checkpoint_path = artifact_dir / "autoencoder_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for maps in train_loader:
            maps = maps.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=scaler_enabled,
            ):
                reconstruction, _ = model(maps)
                loss = F.mse_loss(reconstruction.float(), maps.float())
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * maps.shape[0]
            train_count += maps.shape[0]

        model.eval()
        validation_loss = 0.0
        validation_count = 0
        with torch.inference_mode():
            for maps in validation_loader:
                maps = maps.to(args.device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=scaler_enabled,
                ):
                    reconstruction, _ = model(maps)
                    loss = F.mse_loss(
                        reconstruction.float(), maps.float()
                    )
                validation_loss += float(loss.item()) * maps.shape[0]
                validation_count += maps.shape[0]
        train_mean = train_loss / train_count
        validation_mean = validation_loss / validation_count
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mean,
                "validation_mse": validation_mean,
            }
        )
        if epoch % args.log_every == 0:
            logger.info(
                "epoch %d | train_mse=%.8g | validation_mse=%.8g",
                epoch,
                train_mean,
                validation_mean,
            )
        if validation_mean < best_loss - args.min_delta:
            best_loss = validation_mean
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "map_size": args.map_size,
                    "latent_dim": args.latent_dim,
                    "epoch": epoch,
                    "validation_mse": validation_mean,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    training_summary = {
        "schema_version": 1,
        "architecture_source": "SharePrefill Table 3",
        "map_size": args.map_size,
        "latent_dim": args.latent_dim,
        "loss": "mean_squared_error",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "batch_size": args.batch_size,
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "epochs_completed": len(history),
        "train_indices": training,
        "validation_indices": validation,
        "seed": args.seed,
        "history": history,
    }
    (artifact_dir / "autoencoder_training.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    return training_summary


@torch.inference_mode()
def encode_and_cluster(args: argparse.Namespace) -> Dict[int, Path]:
    artifact_dir = Path(args.artifact_dir)
    manifest = _load_manifest(artifact_dir)
    checkpoint_path = artifact_dir / "autoencoder_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Train autoencoder first: {checkpoint_path}")
    dataset = MeanAttentionMapDataset(
        artifact_dir / "mean_attention_maps",
        num_layers=int(manifest["num_layers"]),
        num_heads=int(manifest["num_heads"]),
    )
    model = SharePrefillAutoencoder(
        map_size=args.map_size, latent_dim=args.latent_dim
    ).to(args.device)
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    latent_rows: List[torch.Tensor] = []
    for maps in DataLoader(dataset, batch_size=1, shuffle=False):
        maps = maps.to(args.device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.device.startswith("cuda"),
        ):
            latent = model.encode(maps)
        latent_rows.append(F.normalize(latent.float(), dim=-1).cpu())
    latents = torch.cat(latent_rows, dim=0).view(
        int(manifest["num_layers"]),
        int(manifest["num_heads"]),
        args.latent_dim,
    )
    np.save(artifact_dir / "normalized_latents.npy", latents.numpy())
    training = json.loads(
        (artifact_dir / "autoencoder_training.json").read_text(
            encoding="utf-8"
        )
    )
    outputs: Dict[int, Path] = {}

    for groups_per_layer in args.groups:
        layers: Dict[str, List[Dict[str, Any]]] = {}
        for layer_idx in range(latents.shape[0]):
            layer_latents = latents[layer_idx].numpy()
            hierarchy = linkage(layer_latents, method="ward")
            labels = (
                cut_tree(hierarchy, n_clusters=[groups_per_layer])
                .reshape(-1)
                .astype(int)
            )
            layer_groups: List[Dict[str, Any]] = []
            for label in range(groups_per_layer):
                members = np.flatnonzero(labels == label).tolist()
                if not members:
                    raise RuntimeError(
                        f"Layer {layer_idx} produced an empty cluster"
                    )
                member_latents = layer_latents[members]
                distances = np.linalg.norm(
                    member_latents[:, None, :] - member_latents[None, :, :],
                    axis=-1,
                )
                local_medoid = int(distances.sum(axis=1).argmin())
                representative = int(members[local_medoid])
                representative_distances = distances[local_medoid]
                layer_groups.append(
                    {
                        "representative": representative,
                        "members": [int(member) for member in members],
                        "mean_latent_distance": float(
                            representative_distances.mean()
                        ),
                        "max_latent_distance": float(
                            representative_distances.max()
                        ),
                    }
                )
            layers[str(layer_idx)] = layer_groups

        config = {
            "schema_version": 1,
            "classification_metric": (
                "shareprefill_attention_map_autoencoder"
            ),
            "num_groups_per_layer": groups_per_layer,
            "num_calibration_samples": manifest["num_samples"],
            "calibration_sample_ids": manifest["sample_ids"],
            "layers": layers,
            "autoencoder": {
                "architecture_source": "SharePrefill Table 3",
                "map_size": args.map_size,
                "latent_dim": args.latent_dim,
                "latent_normalization": "l2",
                "training_loss": training["loss"],
                "optimizer": training["optimizer"],
                "learning_rate": training["learning_rate"],
                "max_epochs": training["max_epochs"],
                "early_stopping_patience": training["patience"],
                "best_epoch": training["best_epoch"],
                "best_validation_mse": training[
                    "best_validation_mse"
                ],
                "attention_map_manifest": str(
                    artifact_dir / "attention_map_manifest.json"
                ),
            },
            "clustering": {
                "paper_original": (
                    "normalized latent + scipy hierarchical fcluster "
                    "with distance threshold 10; clusters smaller than 5 "
                    "become noise"
                ),
                "fixed_k_adaptation": (
                    "Ward linkage + scipy cut_tree(n_clusters=K)"
                ),
                "representative_selection": (
                    "Euclidean medoid in L2-normalized latent space"
                ),
            },
            "model_name_or_path": args.model_name_or_path,
            "data_path": args.data_path,
        }
        output = artifact_dir / (
            f"shareprefill_ae_k{groups_per_layer}_head_groups.json"
        )
        output.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        outputs[groups_per_layer] = output
        logger.info("Saved K=%d groups to %s", groups_per_layer, output)
    return outputs


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    if args.stage in {"collect", "all"}:
        collect_attention_maps(args)
    if args.stage in {"train", "all"}:
        train_autoencoder(args)
    if args.stage in {"cluster", "all"}:
        encode_and_cluster(args)


if __name__ == "__main__":
    main()
