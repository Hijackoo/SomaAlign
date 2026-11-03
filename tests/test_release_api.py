"""Lightweight release checks that do not require data or a checkpoint."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from data.SomaData import SomaData
from data.SomaData2F import SomaData2F
from data.SomaData3F import SomaData3F
from eval_retrieval import (
    all_pair_retrieval,
    cosine_sanity,
    directed_recall,
    fixed_subset_indices,
    load_subset_indices,
    random_subset_indices,
    recall_from_positive_mask,
)
from ImageBind.models.helpers import SelectTextEOTAndProject
from ImageBind.models.multimodal_preprocessors import SimpleTokenizer
from inference import encode, load_checkpoint, normalize_checkpoint_keys, prepare_tensor


class ReleaseApiTests(unittest.TestCase):
    def test_checkpoint_prefix_normalization(self) -> None:
        value = torch.ones(1)
        cleaned = normalize_checkpoint_keys({"module.model.weight": value})
        self.assertEqual(set(cleaned), {"weight"})

    def test_checkpoint_loader_accepts_weights_only_and_rejects_extra_state(self) -> None:
        source = torch.nn.Linear(2, 2)
        target = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            clean_path = Path(directory) / "encoder_only.pt"
            torch.save({"model_state_dict": source.state_dict()}, clean_path)
            load_checkpoint(target, clean_path)
            self.assertTrue(torch.equal(source.weight, target.weight))
            self.assertTrue(torch.equal(source.bias, target.bias))

            extra_path = Path(directory) / "contains_training_state.pt"
            state_with_extra = dict(source.state_dict())
            state_with_extra["optimizer.step"] = torch.tensor(1)
            torch.save({"model_state_dict": state_with_extra}, extra_path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(target, extra_path)

            legacy_path = Path(directory) / "legacy_auxiliary_state.pt"
            legacy_state = dict(source.state_dict())
            legacy_state["stage2_loss_fn.log_k"] = torch.tensor(0.0)
            torch.save({"model_state_dict": legacy_state}, legacy_path)
            load_checkpoint(target, legacy_path, allow_extra_keys=True)

    def test_tensor_shapes(self) -> None:
        cpu = torch.device("cpu")
        self.assertEqual(
            tuple(prepare_tensor(torch.zeros(3, 3, 224, 224), "rgb", cpu).shape),
            (1, 3, 3, 224, 224),
        )
        self.assertEqual(
            tuple(prepare_tensor(torch.zeros(6, 2000), "imu", cpu).shape),
            (1, 6, 2000),
        )
        self.assertEqual(
            tuple(prepare_tensor(torch.zeros(77, dtype=torch.long), "text", cpu).shape),
            (1, 77),
        )

    def test_retrieval_identity(self) -> None:
        features = torch.eye(4)
        self.assertEqual(directed_recall(features, features)["R@1"], 100.0)

    def test_multi_positive_recall_accepts_any_valid_match(self) -> None:
        query = torch.eye(2)
        gallery = torch.tensor([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
        positives = torch.tensor([[False, True, False], [True, False, False]])
        result = recall_from_positive_mask(query, gallery, positives)
        self.assertEqual(result["R@1"], 100.0)

    def test_chunked_retrieval_matches_single_chunk(self) -> None:
        generator = torch.Generator().manual_seed(7)
        query = torch.randn(9, 5, generator=generator)
        gallery = torch.randn(11, 5, generator=generator)
        positives = torch.zeros(9, 11, dtype=torch.bool)
        positives[torch.arange(9), torch.arange(9)] = True
        chunked = recall_from_positive_mask(
            query, gallery, positives, query_batch_size=2
        )
        single = recall_from_positive_mask(
            query, gallery, positives, query_batch_size=99
        )
        self.assertEqual(chunked, single)

    def test_description_level_text_gallery_is_deduplicated(self) -> None:
        features = {
            "rgb": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            "text": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        }
        result = all_pair_retrieval(features, ["same", "same", "other"])
        self.assertEqual(result["protocol"], "description_level_multipositive")
        self.assertEqual(result["unique_descriptions"], 2)
        self.assertEqual(result["tasks"]["rgb->text"]["gallery_count"], 2)
        self.assertEqual(result["tasks"]["text->rgb"]["query_count"], 2)
        self.assertEqual(result["tasks"]["text->rgb"]["positive_count_max"], 2)
        self.assertEqual(result["average"]["R@1"], 100.0)

    def test_random_subset_is_user_configurable_and_reproducible(self) -> None:
        first = random_subset_indices(100, 12, 7)
        second = random_subset_indices(100, 12, 7)
        different = random_subset_indices(100, 12, 8)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertNotEqual(first.tolist(), different.tolist())
        self.assertEqual(len(set(first.tolist())), 12)
        self.assertEqual(fixed_subset_indices(100, 12, 7).tolist(), first.tolist())

    def test_explicit_sample_indices_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.json"
            path.write_text('{"indices": [9, 2, 5]}', encoding="utf-8")
            self.assertEqual(load_subset_indices(path, 10).tolist(), [9, 2, 5])

    def test_cosine_negative_pair_is_not_identity(self) -> None:
        features = torch.eye(2)
        result = cosine_sanity({"a": features, "b": features}, seed=42)["a-b"]
        self.assertEqual(result["positive"], 1.0)
        self.assertEqual(result["negative"], 0.0)

    def test_missing_split_fails_closed(self) -> None:
        instance = object.__new__(SomaData)
        instance.allow_generated_split = False
        dataset = TensorDataset(torch.arange(4))
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                instance._create_split_subset(dataset, str(missing), "test", "2F")

    def test_dataset_directory_order_is_stable(self) -> None:
        for loader_type in (SomaData2F, SomaData3F):
            with self.subTest(loader=loader_type.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for trial, sample in (("2", "0"), ("1", "1"), ("1", "0")):
                    (root / "1_Object" / trial / sample).mkdir(parents=True)
                loader = object.__new__(loader_type)
                loader.root_dir = str(root)
                loader.enabled_modalities = {}
                loader.strict_validation = True
                loader._get_label = lambda *_: 1
                samples = loader._collect_samples_strict()
                self.assertEqual(
                    [(item["trial"], item["sample"]) for item in samples],
                    [("1", "0"), ("1", "1"), ("2", "0")],
                )

    def test_named_sample_split_is_independent_of_dataset_order(self) -> None:
        dataset = TensorDataset(torch.arange(3))
        dataset.samples = [
            {"class": "1_Object", "trial": "2", "sample": "0"},
            {"class": "1_Object", "trial": "1", "sample": "1"},
            {"class": "1_Object", "trial": "1", "sample": "0"},
        ]
        instance = object.__new__(SomaData)
        instance.allow_generated_split = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"sample_ids": {
                "train": ["1_Object/1/0"],
                "val": ["1_Object/2/0"],
                "test": ["1_Object/1/1"],
            }}), encoding="utf-8")
            subset = instance._create_split_subset(dataset, str(path), "test", "2F")
            self.assertEqual(subset.indices, [1])

    def test_user_defined_numeric_split_and_overlap_validation(self) -> None:
        dataset = TensorDataset(torch.arange(4))
        instance = object.__new__(SomaData)
        instance.allow_generated_split = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numeric.json"
            manifest = {"indices": {
                "train_indices": [0, 1],
                "val_indices": [2],
                "test_indices": [3],
            }}
            path.write_text(json.dumps(manifest), encoding="utf-8")
            subset = instance._create_split_subset(dataset, str(path), "test", "2F")
            self.assertEqual(subset.indices, [3])
            manifest["indices"]["test_indices"] = [2]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlapping indices"):
                instance._create_split_subset(dataset, str(path), "test", "2F")

    def test_generated_split_is_explicit_and_reproducible(self) -> None:
        dataset = TensorDataset(torch.arange(100))
        instance = object.__new__(SomaData)
        instance.allow_generated_split = True
        instance.split_seed = 42
        first = instance._create_split_subset(dataset, None, "test", "2F")
        second = instance._create_split_subset(dataset, None, "test", "2F")
        self.assertEqual(first.indices, second.indices)
        self.assertEqual(len(first), 15)
        instance.split_seed = 43
        different = instance._create_split_subset(dataset, None, "test", "2F")
        self.assertNotEqual(first.indices, different.indices)

    def test_text_head_accepts_preprocessor_keyword(self) -> None:
        head = SelectTextEOTAndProject(torch.nn.Identity())
        value = torch.randn(2, 77, 8)
        result = head(value, seq_len=torch.tensor([5, 9]))
        self.assertEqual(tuple(result.shape), (2, 8))
        self.assertTrue(torch.equal(result[0], value[0, 5]))
        self.assertTrue(torch.equal(result[1], value[1, 9]))

    def test_long_text_keeps_end_of_text_token(self) -> None:
        tokenizer = SimpleTokenizer("unused")
        tokens = tokenizer("multimodal " * 200, context_length=77)
        eot = tokenizer.encoder["<|endoftext|>"]
        self.assertEqual(int(tokens[-1]), eot)
        self.assertEqual(int(tokens.argmax()), 76)

    def test_encode_returns_unit_embeddings_after_head_scaling(self) -> None:
        class ScaledModel:
            def __call__(self, inputs, normalize=True):
                key = next(iter(inputs))
                return {key: torch.tensor([[30.0, 40.0]])}

        outputs = encode(
            ScaledModel(),
            {"rgb": torch.zeros(3, 8, 8)},
            torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(outputs["rgb"].norm(dim=1), torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
