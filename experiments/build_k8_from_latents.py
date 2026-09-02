#!/usr/bin/env python3
"""Create a fixed-K SharePrefill group file from saved normalized latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    latents = np.load(artifact_dir / "normalized_latents.npy")
    manifest = json.loads(
        (artifact_dir / "attention_map_manifest.json").read_text(encoding="utf-8")
    )
    training = json.loads(
        (artifact_dir / "autoencoder_training.json").read_text(encoding="utf-8")
    )
    expected_shape = (
        int(manifest["num_layers"]),
        int(manifest["num_heads"]),
        64,
    )
    if latents.shape != expected_shape:
        raise ValueError(f"Expected latent shape {expected_shape}, got {latents.shape}")
    if not 1 <= args.groups <= latents.shape[1]:
        raise ValueError("groups must be between 1 and num_heads")

    layers: dict[str, list[dict]] = {}
    for layer_idx, layer_latents in enumerate(latents):
        labels = (
            cut_tree(linkage(layer_latents, method="ward"), n_clusters=[args.groups])
            .reshape(-1)
            .astype(int)
        )
        layer_groups = []
        for label in range(args.groups):
            members = np.flatnonzero(labels == label).tolist()
            if not members:
                raise RuntimeError(f"Layer {layer_idx} produced an empty cluster")
            member_latents = layer_latents[members]
            distances = np.linalg.norm(
                member_latents[:, None, :] - member_latents[None, :, :], axis=-1
            )
            local_medoid = int(distances.sum(axis=1).argmin())
            representative = int(members[local_medoid])
            representative_distances = distances[local_medoid]
            layer_groups.append(
                {
                    "representative": representative,
                    "members": [int(member) for member in members],
                    "mean_latent_distance": float(representative_distances.mean()),
                    "max_latent_distance": float(representative_distances.max()),
                }
            )
        layers[str(layer_idx)] = layer_groups

    config = {
        "schema_version": 1,
        "classification_metric": "shareprefill_attention_map_autoencoder",
        "num_groups_per_layer": args.groups,
        "num_calibration_samples": manifest["num_samples"],
        "calibration_sample_ids": manifest["sample_ids"],
        "layers": layers,
        "autoencoder": {
            "architecture_source": "SharePrefill Table 3",
            "map_size": int(manifest["map_size"]),
            "latent_dim": int(latents.shape[-1]),
            "latent_normalization": "l2",
            "training_loss": training["loss"],
            "optimizer": training["optimizer"],
            "learning_rate": training["learning_rate"],
            "max_epochs": training["max_epochs"],
            "early_stopping_patience": training["patience"],
            "best_epoch": training["best_epoch"],
            "best_validation_mse": training["best_validation_mse"],
            "attention_map_manifest": str(
                artifact_dir / "attention_map_manifest.json"
            ),
            "normalized_latents": str(artifact_dir / "normalized_latents.npy"),
        },
        "clustering": {
            "fixed_k_adaptation": "Ward linkage + scipy cut_tree(n_clusters=K)",
            "representative_selection": "Euclidean medoid in L2-normalized latent space",
        },
        "model_name_or_path": manifest["model_name_or_path"],
        "data_path": manifest["data_path"],
    }
    output = artifact_dir / f"shareprefill_ae_k{args.groups}_head_groups.json"
    output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
