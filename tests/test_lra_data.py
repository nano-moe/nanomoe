from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from nanomoe.lra.data import classification_collate_fn, load_lra_datasets, resolve_lra_data_root


def test_resolve_lra_data_root_prefers_explicit_path(tmp_path: Path) -> None:
    resolved = resolve_lra_data_root(tmp_path)
    assert resolved == tmp_path.resolve()


def test_load_listops_dataset_from_local_tsv(tmp_path: Path) -> None:
    data_root = tmp_path / "listops"
    data_root.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (data_root / f"basic_{split}.tsv").write_text("Source\tTarget\n[MAX 1 2]\t3\n[MIN 4 5]\t1\n")

    datasets = load_lra_datasets("listops", data_root=tmp_path, max_length=16)
    assert len(datasets.train) == 2
    assert len(datasets.val) == 2
    assert len(datasets.test) == 2
    assert datasets.spec.num_classes == 10
    assert datasets.vocab is not None
    assert datasets.vocab.pad_id == 0

    collate = classification_collate_fn(datasets.pad_value)
    batch = collate([datasets.train[0], datasets.train[1]])
    assert batch.inputs.shape == (2, 4)
    assert batch.attention_mask.dtype == torch.bool
    assert batch.labels.tolist() == [3, 1]


def test_load_pathx_dataset_from_local_tree(tmp_path: Path) -> None:
    data_root = tmp_path / "pathfinder" / "pathfinder128" / "curv_contour_length_14"
    metadata_dir = data_root / "metadata"
    image_dir = data_root / "imgs" / "0"
    metadata_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    for index in range(10):
        image = Image.new("L", (2, 2), color=index)
        image.save(image_dir / f"sample_{index}.png")
    (metadata_dir / "0.npy").write_text("\n".join(f"imgs/0 sample_{index}.png 0 {index % 2}" for index in range(10)))

    datasets = load_lra_datasets("pathx", data_root=tmp_path, max_train_examples=4, max_eval_examples=2, seed=0)
    assert datasets.vocab is None
    assert datasets.spec.input_mode == "continuous"
    assert len(datasets.train) == 4
    assert len(datasets.val) == 1
    assert len(datasets.test) == 1

    batch = classification_collate_fn(datasets.pad_value)([datasets.train[0], datasets.train[1]])
    assert batch.inputs.ndim == 3
    assert batch.inputs.shape[-1] == 1
