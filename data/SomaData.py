"""Unified 2F/3F dataset adapter for SomaAlign evaluation."""

import os
import json
import logging
from torch.utils.data import Dataset, ConcatDataset

from data.SomaData2F import SomaData2F
from data.SomaData3F import SomaData3F

class SomaData(Dataset):
    """
    Unified dataset that combines 2F and 3F sensor data.

    Usage:
        dataset = SomaData(
            data_root_2f="/path/to/2F/data",
            data_root_3f="/path/to/3F/data",
            split_file_2f="/path/to/my_2f_split.json",
            split_file_3f="/path/to/my_3f_split.json",
            split="test",
            ...
        )
    """

    def __init__(self, data_root_2f, data_root_3f, split_file_2f=None, split_file_3f=None, split="train",
                 label_file_2f="标签.xlsx", label_file_3f="标签.xlsx",
                 text_annotations_file_2f="text_annotations.json",
                 text_annotations_file_3f="text_annotations.json",
                 bpe_path=None, window_size=3, strict_validation=True,
                 allow_generated_split=False, split_seed=42):
        """
        Args:
            data_root_2f: 2F sensor data root directory
            data_root_3f: 3F sensor data root directory
            split_file_2f: 2F dataset split file
            split_file_3f: 3F dataset split file
            split: "train", "val", or "test"
            label_file_2f: 2F label file name
            label_file_3f: 3F label file name
            text_annotations_file_2f: 2F text annotation file name
            text_annotations_file_3f: 3F text annotation file name
            bpe_path: BPE file path
            window_size: temporal window size
            strict_validation: whether to enforce strict validation
            allow_generated_split: explicitly permit a seeded 70/15/15 split
                when a split manifest is unavailable
            split_seed: random seed used when generating a split
        """
        self.split = split
        self.window_size = window_size
        self.strict_validation = strict_validation
        self.allow_generated_split = allow_generated_split
        self.split_seed = split_seed

        if not allow_generated_split:
            missing_manifests = [
                path for path in (split_file_2f, split_file_3f)
                if not path or not os.path.isfile(path)
            ]
            if missing_manifests:
                raise FileNotFoundError(
                    "Required split manifest(s) not found: "
                    + ", ".join(str(path) for path in missing_manifests)
                )

        print("=" * 80)
        print(f"SomaData init (2F + 3F mixed)  split={split}  window={window_size}  strict={strict_validation}")
        print("=" * 80)

        # Load 2F dataset
        print("Loading 2F sensor dataset...")
        logging.info("Loading 2F sensor dataset...")
        self.dataset_2f_full = SomaData2F(
            root_dir=data_root_2f,
            label_file=label_file_2f,
            text_annotations_file=text_annotations_file_2f,
            bpe_path=bpe_path,
            window_size=window_size,
            strict_validation=strict_validation
        )
        logging.info(f"2F dataset loaded: {len(self.dataset_2f_full)} samples")

        # Load 3F dataset
        print("Loading 3F sensor dataset...")
        logging.info("Loading 3F sensor dataset...")
        self.dataset_3f_full = SomaData3F(
            root_dir=data_root_3f,
            label_file=label_file_3f,
            text_annotations_file=text_annotations_file_3f,
            bpe_path=bpe_path,
            window_size=window_size,
            strict_validation=strict_validation
        )
        logging.info(f"3F dataset loaded: {len(self.dataset_3f_full)} samples")

        # Apply split
        print(f"Applying split ({split})...")
        self.dataset_2f = self._create_split_subset(
            self.dataset_2f_full, split_file_2f, split, "2F"
        )
        self.dataset_3f = self._create_split_subset(
            self.dataset_3f_full, split_file_3f, split, "3F"
        )

        # Merge
        print("Merging 2F and 3F datasets...")
        logging.info("Merging 2F and 3F datasets...")
        self.combined_dataset = ConcatDataset([self.dataset_2f, self.dataset_3f])
        logging.info(f"Merge complete: {len(self.combined_dataset)} samples")

        self._build_index_mapping()

        print(f"\nUnified dataset ready:")
        print(f"   2F ({split}): {len(self.dataset_2f)}")
        print(f"   3F ({split}): {len(self.dataset_3f)}")
        print(f"   Total: {len(self.combined_dataset)}")
        print("=" * 80)
        print()

    def _auto_split(self, full_dataset, split):
        """Split dataset by 70/15/15 ratio with the user-selected seed."""
        import random
        from torch.utils.data import Subset

        total = len(full_dataset)
        indices = list(range(total))
        random.Random(self.split_seed).shuffle(indices)

        n_train = int(total * 0.7)
        n_val = int(total * 0.15)

        if split == "train":
            split_indices = indices[:n_train]
        elif split == "val":
            split_indices = indices[n_train:n_train + n_val]
        else:
            split_indices = indices[n_train + n_val:]

        return Subset(full_dataset, split_indices)

    def _create_split_subset(self, full_dataset, split_file, split, sensor_type):
        """Create a subset from a manifest; optional fallback is explicit."""
        from torch.utils.data import Subset

        if not split_file or not os.path.isfile(split_file):
            if not self.allow_generated_split:
                raise FileNotFoundError(
                    f"Split file not found ({sensor_type}): {split_file}. "
                    "Provide a split manifest or set allow_generated_split=True "
                    "to use a seeded 70/15/15 split."
                )
            logging.warning(
                "%s split file not found: %s; using 70/15/15 split with seed %s",
                sensor_type,
                split_file,
                self.split_seed,
            )
            print(f"   {sensor_type}: using generated split (70/15/15, seed={self.split_seed})")
            subset = self._auto_split(full_dataset, split)
            print(f"   {sensor_type} {split}: {len(subset)} samples (auto)")
            return subset

        with open(split_file, 'r', encoding='utf-8') as f:
            split_info = json.load(f)

        if not isinstance(split_info, dict):
            raise ValueError(f"Split manifest must be a JSON object: {split_file}")

        # Sample IDs do not depend on directory enumeration order.
        if isinstance(split_info.get('sample_ids'), dict):
            sample_ids = {
                f"{item['class']}/{item['trial']}/{item['sample']}": index
                for index, item in enumerate(full_dataset.samples)
            }
            if len(sample_ids) != len(full_dataset):
                raise ValueError(f"Dataset contains duplicate sample IDs: {split_file}")
            selections = self._validate_named_splits(
                split_info['sample_ids'], ('train', 'val', 'test'), split_file
            )
            missing = set().union(*(set(values) for values in selections.values())) - set(sample_ids)
            if missing:
                raise ValueError(f"Unknown sample ID(s) in {split_file}: {sorted(missing)[:5]}")
            indices = [sample_ids[value] for value in selections[split]]
        elif isinstance(split_info.get('indices'), dict):
            selections = split_info['indices']
            required = ('train_indices', 'val_indices', 'test_indices')
            if any(key not in selections for key in required):
                raise ValueError(f"Split manifest must contain {required}: {split_file}")
            for key in required:
                values = selections[key]
                if not isinstance(values, list) or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    or value < 0 or value >= len(full_dataset) for value in values
                ) or len(values) != len(set(values)):
                    raise ValueError(f"Invalid or duplicate indices in {key}: {split_file}")
            if any(set(selections[left]) & set(selections[right]) for left, right in (
                ('train_indices', 'val_indices'), ('train_indices', 'test_indices'),
                ('val_indices', 'test_indices')
            )):
                raise ValueError(f"Split manifest has overlapping indices: {split_file}")
            indices = selections[f'{split}_indices']
        else:
            raise ValueError(
                f"Split manifest needs sample_ids or indices: {split_file}"
            )

        subset = Subset(full_dataset, indices)

        print(f"   {sensor_type} {split}: {len(indices)} samples")
        return subset

    @staticmethod
    def _validate_named_splits(selections, keys, split_file):
        if any(key not in selections for key in keys):
            raise ValueError(f"Split manifest must contain {keys}: {split_file}")
        for key in keys:
            values = selections[key]
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value for value in values
            ) or len(values) != len(set(values)):
                raise ValueError(f"Invalid or duplicate names in {key}: {split_file}")
        if any(set(selections[left]) & set(selections[right]) for left, right in (
            (keys[0], keys[1]), (keys[0], keys[2]), (keys[1], keys[2])
        )):
            raise ValueError(f"Split manifest has overlapping names: {split_file}")
        return selections

    def _build_index_mapping(self):
        """Build global-to-local index mapping for the ConcatDataset."""
        self.index_mapping = []
        self.cumulative_sizes = self.combined_dataset.cumulative_sizes

        global_idx = 0
        for dataset_idx, dataset in enumerate([self.dataset_2f, self.dataset_3f]):
            for local_idx in range(len(dataset)):
                self.index_mapping.append({
                    'dataset_idx': dataset_idx,   # 0=2F, 1=3F
                    'local_idx': local_idx,
                    'sensor_type': '2F' if dataset_idx == 0 else '3F',
                    'global_idx': global_idx
                })
                global_idx += 1

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        return self.combined_dataset[idx]

    def get_sensor_type(self, idx):
        """Return sensor type ('2F' or '3F') for the given index."""
        if idx < len(self.dataset_2f):
            return '2F'
        else:
            return '3F'

    def get_modality_list(self):
        """Return list of enabled modalities."""
        return self.dataset_2f_full.get_modality_list()

    def validate_all_samples(self, num_samples=None):
        """Validate all samples by attempting to load each one."""
        print("Pre-validating unified dataset...")

        num_to_validate = num_samples if num_samples else len(self.combined_dataset)
        invalid_samples = []

        for idx in range(min(num_to_validate, len(self.combined_dataset))):
            try:
                self[idx]
                sensor_type = self.get_sensor_type(idx)
                print(f"   Sample {idx} ({sensor_type}): OK")
            except Exception as e:
                sensor_type = self.get_sensor_type(idx)
                invalid_samples.append((idx, sensor_type, str(e)))
                print(f"   Sample {idx} ({sensor_type}): FAILED - {str(e)}")

        print(f"\nValidation result:")
        print(f"   Valid: {num_to_validate - len(invalid_samples)}")
        print(f"   Invalid: {len(invalid_samples)}")

        if invalid_samples:
            print("\nInvalid samples:")
            for idx, sensor_type, error in invalid_samples[:5]:
                print(f"   Sample {idx} ({sensor_type}): {error}")
            if len(invalid_samples) > 5:
                print(f"   ... ({len(invalid_samples)} total invalid samples)")

        return invalid_samples


def create_unified_datasets(
    data_root_2f,
    data_root_3f,
    split_file_2f=None,
    split_file_3f=None,
    label_file_2f="标签.xlsx",
    label_file_3f="标签.xlsx",
    text_annotations_file_2f="text_annotations.json",
    text_annotations_file_3f="text_annotations.json",
    bpe_path=None,
    window_size=3,
    strict_validation=True,
    allow_generated_split=False,
    split_seed=42,
):
    """Create train, validation, and test adapters for API compatibility.

    This is a dataset convenience interface only; it does not implement a
    trainer, loss, optimizer, or any other training procedure.
    """
    common = dict(
        data_root_2f=data_root_2f,
        data_root_3f=data_root_3f,
        split_file_2f=split_file_2f,
        split_file_3f=split_file_3f,
        label_file_2f=label_file_2f,
        label_file_3f=label_file_3f,
        text_annotations_file_2f=text_annotations_file_2f,
        text_annotations_file_3f=text_annotations_file_3f,
        bpe_path=bpe_path,
        window_size=window_size,
        strict_validation=strict_validation,
        allow_generated_split=allow_generated_split,
        split_seed=split_seed,
    )
    return tuple(SomaData(split=split, **common) for split in ("train", "val", "test"))
