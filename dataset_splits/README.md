# Dataset split files

Pass split manifests to `eval_retrieval.py` with `--split_file_2f` and
`--split_file_3f`.

For portable, user-defined splits, use sample IDs of the form
`<class>/<trial>/<sample>`:

```json
{
  "sample_ids": {
    "train": ["1_Object/1/0"],
    "val": ["1_Object/2/0"],
    "test": ["1_Object/3/0"]
  }
}
```

You may instead provide `indices.train_indices`, `indices.val_indices`, and
`indices.test_indices`. These are zero-based positions in the loader's stable
lexicographic `<class>/<trial>/<sample>` order, and must be created for the
same dataset and modality configuration.

All named or numeric splits reject duplicates, overlap, and unknown sample
IDs or out-of-range indices. To generate a 70/15/15 sample-level split instead,
pass `--generate_split` and choose the random seed with `--seed`. This generated
split is reproducible for the same dataset, enabled modalities, and seed.
It is a convenience split for running the example, not a prescribed paper
evaluation split. If trial-level separation matters for your evaluation,
create manifests with whole trials assigned to only one partition.
