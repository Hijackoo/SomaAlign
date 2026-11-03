#!/usr/bin/env python3
"""SomaAlign cross-modal retrieval evaluation.

Loads an encoder checkpoint, extracts L2-normalized embeddings for all six
modality channels, and evaluates paired and description-level retrieval:

* sensory-to-sensory retrieval uses paired observation-level positives;
* sensory-to-text uses one gallery item per unique structured description;
* text-to-sensory uses one query per unique description and every sensory
  observation with that exact description as a valid positive.

The evaluator does not prescribe a fixed subset. By default it evaluates
the complete user-selected split; callers may instead request a seeded random
subset or provide an explicit JSON list of split-relative indices.

Usage:
    python eval_retrieval.py \
        --checkpoint best_model.pth \
        --data_root_2f /path/to/DataOrigin2FProcess \
        --data_root_3f /path/to/DataOrigin3FProcess \
        --output results/retrieval.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.SomaData import SomaData
from inference import build_model, encode, load_checkpoint

NON_TEXT_MODALITIES = ["rgb", "photo", "depth", "tactile", "imu"]
DEFAULT_RETRIEVAL_QUERY_BATCH_SIZE = 512
# ---------------------------------------------------------------- metrics

def _validate_retrieval_inputs(
    query: torch.Tensor,
    target: torch.Tensor,
    ks: tuple[int, ...],
    query_batch_size: int,
) -> int:
    if query.ndim != 2 or target.ndim != 2:
        raise ValueError("query and gallery embeddings must be two-dimensional")
    if query.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("query and gallery embeddings cannot be empty")
    if query.shape[1] != target.shape[1]:
        raise ValueError("query and gallery embedding dimensions must match")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    return min(max(ks), target.shape[0])


def directed_recall(
    query: torch.Tensor,
    target: torch.Tensor,
    ks=(1, 5, 10),
    query_batch_size: int = DEFAULT_RETRIEVAL_QUERY_BATCH_SIZE,
) -> dict:
    """Recall@K for directed retrieval: row i of `query` must retrieve row i of `target`.

    Assumes query.shape[0] == target.shape[0] (paired samples).
    """
    if query.shape[0] != target.shape[0]:
        raise ValueError("paired retrieval requires equal query and gallery counts")
    labels = torch.arange(query.shape[0])
    return recall_from_labels(
        query, target, labels, labels, ks=ks, query_batch_size=query_batch_size
    )


def recall_from_positive_mask(
    query: torch.Tensor,
    target: torch.Tensor,
    positive_mask: torch.Tensor,
    ks=(1, 5, 10),
    query_batch_size: int = DEFAULT_RETRIEVAL_QUERY_BATCH_SIZE,
) -> dict:
    """Recall@K with arbitrary positives, evaluated in bounded query chunks."""
    expected = (query.shape[0], target.shape[0])
    if tuple(positive_mask.shape) != expected:
        raise ValueError(
            f"positive mask must have shape {expected}, got {tuple(positive_mask.shape)}"
        )
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("every query must have at least one positive")
    ks = tuple(ks)
    max_k = _validate_retrieval_inputs(query, target, ks, query_batch_size)
    unit_query = F.normalize(query, dim=1)
    unit_target = F.normalize(target, dim=1)
    hits = {k: 0 for k in ks}
    for start in range(0, query.shape[0], query_batch_size):
        stop = min(start + query_batch_size, query.shape[0])
        similarity = unit_query[start:stop] @ unit_target.T
        ranking = torch.argsort(
            similarity, dim=1, descending=True, stable=True
        )[:, :max_k]
        mask = positive_mask[start:stop].to(device=query.device, dtype=torch.bool)
        for k in ks:
            hits[k] += int(mask.gather(1, ranking[:, : min(k, max_k)]).any(dim=1).sum())
    return {
        f"R@{k}": float(
            torch.tensor(hits[k] / query.shape[0], dtype=torch.float32).item() * 100
        )
        for k in ks
    }


def recall_from_labels(
    query: torch.Tensor,
    target: torch.Tensor,
    query_labels: torch.Tensor,
    target_labels: torch.Tensor,
    ks=(1, 5, 10),
    query_batch_size: int = DEFAULT_RETRIEVAL_QUERY_BATCH_SIZE,
) -> dict:
    """Recall@K where equal integer labels identify valid positives.

    This representation avoids allocating a dense query-by-gallery positive
    mask for paired and description-level retrieval.
    """
    ks = tuple(ks)
    max_k = _validate_retrieval_inputs(query, target, ks, query_batch_size)
    query_labels = torch.as_tensor(query_labels).reshape(-1)
    target_labels = torch.as_tensor(target_labels).reshape(-1)
    if query_labels.shape[0] != query.shape[0] or target_labels.shape[0] != target.shape[0]:
        raise ValueError("there must be one label per query and gallery embedding")
    if not torch.isin(query_labels.cpu(), target_labels.cpu()).all():
        raise ValueError("every query label must have at least one gallery positive")

    unit_query = F.normalize(query, dim=1)
    unit_target = F.normalize(target, dim=1)
    target_labels = target_labels.to(query.device)
    hits = {k: 0 for k in ks}
    for start in range(0, query.shape[0], query_batch_size):
        stop = min(start + query_batch_size, query.shape[0])
        similarity = unit_query[start:stop] @ unit_target.T
        ranking = torch.argsort(
            similarity, dim=1, descending=True, stable=True
        )[:, :max_k]
        ranked_labels = target_labels[ranking]
        expected = query_labels[start:stop].to(query.device)[:, None]
        for k in ks:
            hits[k] += int((ranked_labels[:, : min(k, max_k)] == expected).any(dim=1).sum())
    return {
        f"R@{k}": float(
            torch.tensor(hits[k] / query.shape[0], dtype=torch.float32).item() * 100
        )
        for k in ks
    }


def all_pair_retrieval(
    features: dict,
    descriptions: list[str] | None = None,
    query_batch_size: int = DEFAULT_RETRIEVAL_QUERY_BATCH_SIZE,
) -> dict:
    """Evaluate all directions with description-level positive sets."""
    if not features:
        raise ValueError("features cannot be empty")
    sample_count = next(iter(features.values())).shape[0]
    if any(value.shape[0] != sample_count for value in features.values()):
        raise ValueError("all modality feature tensors must have equal sample counts")

    has_text = "text" in features
    if has_text:
        if descriptions is None or len(descriptions) != sample_count:
            raise ValueError("one structured description is required per sample")
        # Python dictionaries preserve insertion order, so the first occurrence
        # is the deterministic representative used by the paper evaluator.
        unique_descriptions = list(dict.fromkeys(descriptions))
        group_by_description = {value: index for index, value in enumerate(unique_descriptions)}
        group_of_sample = torch.tensor(
            [group_by_description[value] for value in descriptions], dtype=torch.long
        )
        representatives = torch.tensor(
            [descriptions.index(value) for value in unique_descriptions], dtype=torch.long
        )

    sample_labels = torch.arange(sample_count)
    tasks = {}
    for src in features:
        for tgt in features:
            if src != tgt:
                query = features[src]
                gallery = features[tgt]
                text_related = src == "text" or tgt == "text"
                if not text_related:
                    query_labels = sample_labels
                    target_labels = sample_labels
                elif src != "text" and tgt == "text":
                    gallery = gallery[representatives]
                    query_labels = group_of_sample
                    target_labels = torch.arange(len(unique_descriptions))
                else:  # Text -> sensory
                    query = query[representatives]
                    query_labels = torch.arange(len(unique_descriptions))
                    target_labels = group_of_sample

                metrics = recall_from_labels(
                    query,
                    gallery,
                    query_labels,
                    target_labels,
                    query_batch_size=query_batch_size,
                )
                target_counts = torch.bincount(target_labels)
                positive_counts = target_counts[query_labels].float()
                tasks[f"{src}->{tgt}"] = {
                    **metrics,
                    "query_count": int(query.shape[0]),
                    "gallery_count": int(gallery.shape[0]),
                    "positive_count_mean": float(positive_counts.mean()),
                    "positive_count_min": int(positive_counts.min()),
                    "positive_count_max": int(positive_counts.max()),
                    "text_related": text_related,
                }
    averages = {
        key: float(np.mean([task[key] for task in tasks.values()]))
        for key in ("R@1", "R@5", "R@10")
    }
    result = {
        "protocol": "description_level_multipositive",
        "multi_positive_hit_rule": "any(topK intersects positive_set)",
        "tasks": tasks,
        "average": averages,
    }
    if has_text:
        result["unique_descriptions"] = len(unique_descriptions)
        result["description_equality"] = "exact structured-description string equality"
    return result


def random_subset_indices(dataset_size: int, sample_count: int, seed: int) -> np.ndarray:
    """Select a reproducible random subset from a user-provided split."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if sample_count > dataset_size:
        raise ValueError(f"sample_count {sample_count} exceeds dataset size {dataset_size}")
    return np.random.RandomState(seed).permutation(dataset_size)[:sample_count]


def fixed_subset_indices(dataset_size: int, sample_count: int, seed: int) -> np.ndarray:
    """Backward-compatible alias for :func:`random_subset_indices`."""
    return random_subset_indices(dataset_size, sample_count, seed)


def load_subset_indices(path: str | Path, dataset_size: int) -> np.ndarray:
    """Load and validate split-relative indices from a JSON list or object."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("indices")
    if not isinstance(payload, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in payload
    ):
        raise ValueError("sample index file must be a JSON list of integers or {'indices': [...]}")
    if len(payload) < 2:
        raise ValueError("evaluation requires at least two sample indices")
    if len(payload) != len(set(payload)):
        raise ValueError("sample index file contains duplicate indices")
    if any(value < 0 or value >= dataset_size for value in payload):
        raise ValueError(f"sample index file contains indices outside [0, {dataset_size})")
    return np.asarray(payload, dtype=int)


def cosine_sanity(features: dict, seed: int = 42) -> dict:
    """Mean cosine of matched pairs vs randomly permuted pairs, per modality pair.

    Inputs are L2-normalized internally, so values are true cosines in [-1, 1]
    regardless of upstream normalization (idempotent for unit inputs).

    Requires at least two modalities and equal sample counts per modality.
    """
    if len(features) < 2:
        raise ValueError("cosine_sanity requires at least two modalities")
    values = {}
    reference = next(iter(features.values()))
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    count = reference.shape[0]
    if count < 2:
        raise ValueError("cosine_sanity requires at least two paired samples")
    # A non-zero cyclic shift guarantees that no negative pair is a positive
    # pair, including deterministic smoke tests with only two samples.
    shift = int(torch.randint(1, count, (1,), generator=generator, device=reference.device))
    permutation = torch.roll(torch.arange(count, device=reference.device), shifts=shift)
    unit = {name: F.normalize(features[name], dim=1) for name in features}
    names = list(unit)
    for i, src in enumerate(names):
        for tgt in names[i + 1:]:
            positive = (unit[src] * unit[tgt]).sum(dim=1).mean().item()
            negative = (unit[src] * unit[tgt][permutation]).sum(dim=1).mean().item()
            values[f"{src}-{tgt}"] = {"positive": positive, "negative": negative}
    return values


# ---------------------------------------------------------------- features


@torch.no_grad()
def extract_all_features(
    model: ImageBindModelForAlignment,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    modalities: list[str],
) -> tuple:
    """Single pass over the dataset; per batch, one forward per modality (as in training).

    Returns (features, sample_ids, descriptions) where features maps each
    modality name to an [N, D] float32 CPU tensor of normalized embeddings.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        pin_memory=True,
    )
    collected = {modality: [] for modality in modalities}
    sample_ids = []
    descriptions = []
    for batch in tqdm(loader, desc="extract"):
        samples = [item[0] for item in batch]  # (data_dict, label, text_desc)
        sample_ids.extend(str(sample["sample_id"]) for sample in samples)
        descriptions.extend(str(item[2]) for item in batch)
        for modality in modalities:
            tensors = [sample[modality] for sample in samples]
            batch_tensor = torch.stack(tensors, dim=0)
            features = encode(model, {modality: batch_tensor}, device)
            collected[modality].append(features[modality].cpu())
    features = {modality: torch.cat(chunks, dim=0) for modality, chunks in collected.items()}
    return features, sample_ids, descriptions


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root_2f", required=True)
    parser.add_argument("--data_root_3f", required=True)
    parser.add_argument("--split_file_2f", help="path to a user-defined 2F split manifest")
    parser.add_argument("--split_file_3f", help="path to a user-defined 3F split manifest")
    parser.add_argument(
        "--generate_split", action="store_true",
        help="generate a seeded 70/15/15 sample-level split instead of supplying manifests",
    )
    parser.add_argument("--label_file_2f", default="标签.xlsx")
    parser.add_argument("--label_file_3f", default="标签.xlsx")
    parser.add_argument("--text_annotations_file_2f", default="text_annotations.json")
    parser.add_argument("--text_annotations_file_3f", default="text_annotations.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="results/retrieval.json")
    parser.add_argument("--batch_size", type=int, default=72)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--retrieval_query_batch_size",
        type=int,
        default=DEFAULT_RETRIEVAL_QUERY_BATCH_SIZE,
        help="number of query rows per similarity/ranking chunk (default: 512)",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=None,
        help="optional seeded-random subset size; omit to evaluate the full selected split",
    )
    parser.add_argument(
        "--sample_indices_file",
        help="optional JSON list (or {'indices': [...]}) of split-relative sample indices",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="deprecated alias for --sample_count",
    )
    parser.add_argument(
        "--all_samples",
        action="store_true",
        help="evaluate the complete selected split (default; retained for compatibility)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        "--include_text",
        dest="include_text",
        action="store_true",
        help="evaluate all six modalities (default)",
    )
    text_group.add_argument(
        "--exclude_text",
        dest="include_text",
        action="store_false",
        help="evaluate only the five non-text sensor channels",
    )
    parser.set_defaults(include_text=True)
    parser.add_argument(
        "--legacy_full_load",
        action="store_true",
        help="load a trusted historical checkpoint and discard known auxiliary state",
    )
    args = parser.parse_args()

    subset_options = sum(
        option is not None for option in (args.sample_count, args.sample_indices_file, args.limit)
    )
    if subset_options > 1:
        parser.error("use only one of --sample_count, --sample_indices_file, or --limit")
    if args.all_samples and subset_options:
        parser.error("--all_samples cannot be combined with a subset option")
    requested_count = args.limit if args.limit is not None else args.sample_count
    if requested_count is not None and requested_count < 2:
        parser.error("evaluation requires at least two samples")
    if args.retrieval_query_batch_size <= 0:
        parser.error("--retrieval_query_batch_size must be positive")
    if args.generate_split:
        if args.split_file_2f or args.split_file_3f:
            parser.error("--generate_split cannot be combined with split files")
    elif not args.split_file_2f or not args.split_file_3f:
        parser.error("provide both split files or use --generate_split")
    required_files = {"checkpoint": args.checkpoint}
    if not args.generate_split:
        required_files.update({
            "2F split manifest": args.split_file_2f,
            "3F split manifest": args.split_file_3f,
        })
    for label, path in required_files.items():
        if not Path(path).is_file():
            parser.error(f"{label} not found: {path}")
    for label, path in (("2F data root", args.data_root_2f), ("3F data root", args.data_root_3f)):
        if not Path(path).is_dir():
            parser.error(f"{label} not found: {path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    modalities = NON_TEXT_MODALITIES + (["text"] if args.include_text else [])

    device = torch.device(args.device)
    model = build_model(device)
    load_checkpoint(
        model,
        args.checkpoint,
        weights_only=not args.legacy_full_load,
        allow_extra_keys=args.legacy_full_load,
    )

    dataset = SomaData(
        data_root_2f=args.data_root_2f,
        data_root_3f=args.data_root_3f,
        split_file_2f=None if args.generate_split else args.split_file_2f,
        split_file_3f=None if args.generate_split else args.split_file_3f,
        allow_generated_split=args.generate_split,
        split_seed=args.seed,
        split=args.split,
        label_file_2f=args.label_file_2f,
        label_file_3f=args.label_file_3f,
        text_annotations_file_2f=args.text_annotations_file_2f,
        text_annotations_file_3f=args.text_annotations_file_3f,
    )

    enabled = set(dataset.get_modality_list())
    missing = [m for m in modalities if m not in enabled]
    if missing:
        raise RuntimeError(
            f"modal_config.py disabled modalities required by eval: {missing}; "
            "enable them in modal_config.py to run this protocol"
        )

    if len(dataset) == 0:
        raise ValueError(f"dataset is empty (split={args.split!r})")

    split_size = len(dataset)
    if requested_count is not None and requested_count > split_size:
        parser.error(
            f"requested subset size {requested_count} exceeds selected split size {split_size}"
        )
    if args.sample_indices_file is not None:
        try:
            selected_indices = load_subset_indices(args.sample_indices_file, split_size)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"invalid --sample_indices_file: {error}")
        selection_method = "explicit_json_indices"
        dataset = Subset(dataset, selected_indices.tolist())
    elif requested_count is not None:
        selected_indices = random_subset_indices(split_size, requested_count, args.seed)
        selection_method = "seeded_random_subset"
        dataset = Subset(dataset, selected_indices.tolist())
    else:
        selected_indices = np.arange(split_size, dtype=int)
        selection_method = "all_samples"

    features, sample_ids, descriptions = extract_all_features(
        model, dataset, args.batch_size, args.num_workers, device, modalities
    )

    counts = {modality: features[modality].shape[0] for modality in modalities}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"modality sample counts differ: {counts}")
    num_samples = len(sample_ids)
    print(f"\nSamples: {num_samples}  Embedding dim: {features[modalities[0]].shape[1]}")

    retrieval = all_pair_retrieval(
        features,
        descriptions if args.include_text else None,
        query_batch_size=args.retrieval_query_batch_size,
    )
    sanity = cosine_sanity(features, seed=args.seed)

    print("\nDirected retrieval R@1 / R@5 / R@10:")
    for task, metrics in retrieval["tasks"].items():
        print(f"  {task:<16} {metrics['R@1']:6.2f} {metrics['R@5']:6.2f} {metrics['R@10']:6.2f}")
    avg = retrieval["average"]
    print(f"  {'average':<16} {avg['R@1']:6.2f} {avg['R@5']:6.2f} {avg['R@10']:6.2f}")

    print("\nCosine sanity (positive vs negative):")
    for pair, values in sanity.items():
        flag = "OK" if values["positive"] > values["negative"] else "FAIL"
        print(f"  {pair:<16} {values['positive']:+.4f} vs {values['negative']:+.4f}  [{flag}]")

    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "split_source": "generated_70_15_15" if args.generate_split else "user_manifest",
        "seed": args.seed,
        "num_samples": num_samples,
        "selection": {
            "method": selection_method,
            "dataset_size": split_size,
            "selected_sample_count": num_samples,
            "global_split_indices": selected_indices.tolist(),
        },
        "modalities": modalities,
        "retrieval_query_batch_size": args.retrieval_query_batch_size,
        "retrieval": retrieval,
        "cosine_sanity": sanity,
        "sample_ids": sample_ids,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
