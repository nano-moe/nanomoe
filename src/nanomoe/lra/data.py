from __future__ import annotations

import csv
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

TaskName = Literal["listops", "imdb", "pathx"]

LISTOPS_TRANSLATION = {ord("]"): ord("X"), ord("("): None, ord(")"): None}
PATHFINDER_BLACKLIST = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}


def listops_tokenizer(text: str) -> list[str]:
    return text.translate(LISTOPS_TRANSLATION).split()


def char_tokenizer(text: str) -> list[str]:
    return list(text)


@dataclass(slots=True)
class LRATaskSpec:
    name: TaskName
    num_classes: int
    default_max_length: int
    input_mode: Literal["token", "continuous"]
    append_bos: bool
    append_eos: bool
    min_freq: int = 1
    input_dim: int = 1


TASK_SPECS: dict[TaskName, LRATaskSpec] = {
    "listops": LRATaskSpec(
        name="listops",
        num_classes=10,
        default_max_length=2048,
        input_mode="token",
        append_bos=False,
        append_eos=True,
    ),
    "imdb": LRATaskSpec(
        name="imdb",
        num_classes=2,
        default_max_length=4096,
        input_mode="token",
        append_bos=False,
        append_eos=True,
        min_freq=15,
    ),
    "pathx": LRATaskSpec(
        name="pathx",
        num_classes=2,
        default_max_length=128 * 128,
        input_mode="continuous",
        append_bos=False,
        append_eos=False,
        input_dim=1,
    ),
}


@dataclass(slots=True)
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        unk_id = self.unk_id
        return [self.stoi.get(token, unk_id) for token in tokens]


@dataclass(slots=True)
class ClassificationExample:
    inputs: Tensor
    label: int


@dataclass(slots=True)
class ClassificationBatch:
    inputs: Tensor
    attention_mask: Tensor
    labels: Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "ClassificationBatch":
        return ClassificationBatch(
            inputs=self.inputs.to(device=device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device=device, non_blocking=non_blocking),
            labels=self.labels.to(device=device, non_blocking=non_blocking),
        )

    @property
    def num_tokens(self) -> int:
        return int(self.attention_mask.sum().item())


class ClassificationDataset(Dataset[ClassificationExample]):
    def __init__(self, examples: list[ClassificationExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ClassificationExample:
        return self.examples[index]


def classification_collate_fn(pad_value: int | float):
    def _collate(examples: list[ClassificationExample]) -> ClassificationBatch:
        first = examples[0].inputs
        max_len = max(int(example.inputs.shape[0]) for example in examples)
        inputs = torch.full((len(examples), max_len, *first.shape[1:]), pad_value, dtype=first.dtype)
        attention_mask = torch.zeros((len(examples), max_len), dtype=torch.bool)
        labels = torch.empty(len(examples), dtype=torch.long)
        for row, example in enumerate(examples):
            length = int(example.inputs.shape[0])
            inputs[row, :length] = example.inputs
            attention_mask[row, :length] = True
            labels[row] = example.label
        return ClassificationBatch(inputs=inputs, attention_mask=attention_mask, labels=labels)

    return _collate


def resolve_lra_data_root(data_root: str | os.PathLike[str] | None = None) -> Path:
    candidates: list[Path] = []
    if data_root is not None:
        candidates.append(Path(data_root).expanduser().resolve())
    env_root = os.getenv("NANOMOE_LRA_DATA")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    repo_root = Path(__file__).resolve().parents[3]
    candidates.append((repo_root / "data").resolve())
    candidates.append((repo_root.parent / "s4" / "data").resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def build_vocab(
    token_sequences: Iterable[Iterable[str]],
    *,
    min_freq: int,
    append_bos: bool,
    append_eos: bool,
) -> Vocab:
    counter: Counter[str] = Counter()
    for tokens in token_sequences:
        counter.update(tokens)

    itos = ["<pad>", "<unk>"]
    if append_bos:
        itos.append("<bos>")
    if append_eos:
        itos.append("<eos>")

    kept_tokens = sorted(token for token, freq in counter.items() if freq >= min_freq)
    itos.extend(kept_tokens)
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def _truncate_tokens(tokens: list[str], max_length: int, append_bos: bool, append_eos: bool) -> list[str]:
    usable_length = max_length - int(append_bos) - int(append_eos)
    return tokens[:usable_length]


def _encode_tokens(tokens: list[str], vocab: Vocab, append_bos: bool, append_eos: bool) -> Tensor:
    tokens_out = list(tokens)
    if append_bos:
        tokens_out.insert(0, "<bos>")
    if append_eos:
        tokens_out.append("<eos>")
    return torch.tensor(vocab.encode(tokens_out), dtype=torch.long)


def _load_listops_rows(data_dir: Path, split: str) -> list[tuple[str, int]]:
    split_path = data_dir / f"basic_{split}.tsv"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"ListOps split not found at {split_path}. "
            "Point --data-root at the extracted long-range-arena listops directory."
        )

    rows: list[tuple[str, int]] = []
    with open(split_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append((row["Source"], int(row["Target"])))
    return rows


def _load_imdb_rows(data_root: Path) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    cache_dir = data_root / "imdb"
    dataset = load_dataset("imdb", cache_dir=str(cache_dir))
    train_rows = [(row["text"], int(row["label"])) for row in dataset["train"]]
    test_rows = [(row["text"], int(row["label"])) for row in dataset["test"]]
    return train_rows, test_rows


def _maybe_limit_rows(rows: list[tuple[str, int]], max_examples: int | None) -> list[tuple[str, int]]:
    if max_examples is None:
        return rows
    return rows[:max_examples]


def _tokenize_rows(
    rows: list[tuple[str, int]],
    tokenizer,
    *,
    max_length: int,
    append_bos: bool,
    append_eos: bool,
) -> list[tuple[list[str], int]]:
    tokenized: list[tuple[list[str], int]] = []
    for text, label in rows:
        tokens = tokenizer(text)
        tokens = _truncate_tokens(tokens, max_length, append_bos, append_eos)
        tokenized.append((tokens, label))
    return tokenized


def _materialize_token_examples(
    tokenized_rows: list[tuple[list[str], int]],
    vocab: Vocab,
    *,
    append_bos: bool,
    append_eos: bool,
) -> list[ClassificationExample]:
    return [
        ClassificationExample(
            inputs=_encode_tokens(tokens, vocab, append_bos, append_eos),
            label=label,
        )
        for tokens, label in tokenized_rows
    ]


class PathFinderSequenceDataset(Dataset[ClassificationExample]):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.samples = self._discover_samples(data_dir)

    def _discover_samples(self, data_dir: Path) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        diff_level = "curv_contour_length_14"
        metadata_dir = data_dir / diff_level / "metadata"
        metadata_files = sorted(metadata_dir.glob("*.npy"), key=lambda path: int(path.stem))
        if not metadata_files:
            raise FileNotFoundError(f"No Pathfinder metadata found under {metadata_dir}")

        for metadata_file in metadata_files:
            for fields in self._load_metadata_rows(metadata_file):
                if len(fields) < 4:
                    continue
                image_path = Path(diff_level) / fields[0] / fields[1]
                blacklist_key = str(Path(data_dir.stem) / image_path)
                if blacklist_key in PATHFINDER_BLACKLIST:
                    continue
                samples.append((image_path, int(fields[3])))
        return samples

    @staticmethod
    def _load_metadata_rows(metadata_file: Path) -> list[list[str]]:
        try:
            array = np.load(metadata_file, allow_pickle=True)
        except ValueError:
            array = None
        except Exception:
            array = None

        if array is not None:
            if array.ndim == 1:
                return [str(row).split() for row in array.tolist()]
            return [[str(field) for field in row] for row in array.tolist()]

        with open(metadata_file, encoding="utf-8") as handle:
            return [line.split() for line in handle]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ClassificationExample:
        image_path, label = self.samples[index]
        with open(self.data_dir / image_path, "rb") as handle:
            image = Image.open(handle).convert("L")
            pixels = (
                torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
                .to(dtype=torch.float32)
                .unsqueeze(-1)
                / 255.0
            )
        pixels = (pixels - 0.5) / 0.5
        return ClassificationExample(inputs=pixels, label=label)


def _resolve_pathfinder_root(data_root: Path, resolution: int) -> Path:
    candidates = [
        data_root / "pathfinder" / f"pathfinder{resolution}",
        data_root / f"pathfinder{resolution}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _slice_examples(
    examples: list[ClassificationExample],
    *,
    train: bool,
    max_train_examples: int | None,
    max_eval_examples: int | None,
) -> list[ClassificationExample]:
    limit = max_train_examples if train else max_eval_examples
    if limit is None:
        return examples
    return examples[:limit]


@dataclass(slots=True)
class LRADatasets:
    spec: LRATaskSpec
    vocab: Vocab | None
    pad_value: int | float
    train: ClassificationDataset
    val: ClassificationDataset
    test: ClassificationDataset


def load_lra_datasets(
    task: TaskName,
    *,
    data_root: str | os.PathLike[str] | None = None,
    max_length: int | None = None,
    imdb_val_split: float = 0.0,
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
    seed: int = 42,
) -> LRADatasets:
    spec = TASK_SPECS[task]
    resolved_max_length = max_length or spec.default_max_length
    resolved_root = resolve_lra_data_root(data_root)

    if task == "listops":
        task_root = resolved_root / "listops"
        train_rows = _maybe_limit_rows(_load_listops_rows(task_root, "train"), max_train_examples)
        val_rows = _maybe_limit_rows(_load_listops_rows(task_root, "val"), max_eval_examples)
        test_rows = _maybe_limit_rows(_load_listops_rows(task_root, "test"), max_eval_examples)
        tokenizer = listops_tokenizer
    elif task == "imdb":
        train_rows, test_rows = _load_imdb_rows(resolved_root)
        train_rows = _maybe_limit_rows(train_rows, max_train_examples)
        test_rows = _maybe_limit_rows(test_rows, max_eval_examples)
        tokenizer = char_tokenizer
        if imdb_val_split > 0.0:
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(len(train_rows), generator=generator).tolist()
            val_size = max(1, int(round(imdb_val_split * len(train_rows))))
            val_indices = set(permutation[:val_size])
            val_rows = [row for idx, row in enumerate(train_rows) if idx in val_indices]
            train_rows = [row for idx, row in enumerate(train_rows) if idx not in val_indices]
        else:
            val_rows = test_rows
    elif task == "pathx":
        task_root = _resolve_pathfinder_root(resolved_root, 128)
        full_dataset = PathFinderSequenceDataset(task_root)
        generator = torch.Generator().manual_seed(seed)
        train_len = int(round(0.8 * len(full_dataset)))
        val_len = int(round(0.1 * len(full_dataset)))
        test_len = len(full_dataset) - train_len - val_len
        train_subset, val_subset, test_subset = torch.utils.data.random_split(
            full_dataset,
            [train_len, val_len, test_len],
            generator=generator,
        )
        train_examples = _slice_examples(
            [train_subset[idx] for idx in range(len(train_subset))],
            train=True,
            max_train_examples=max_train_examples,
            max_eval_examples=max_eval_examples,
        )
        val_examples = _slice_examples(
            [val_subset[idx] for idx in range(len(val_subset))],
            train=False,
            max_train_examples=max_train_examples,
            max_eval_examples=max_eval_examples,
        )
        test_examples = _slice_examples(
            [test_subset[idx] for idx in range(len(test_subset))],
            train=False,
            max_train_examples=max_train_examples,
            max_eval_examples=max_eval_examples,
        )
        return LRADatasets(
            spec=spec,
            vocab=None,
            pad_value=0.0,
            train=ClassificationDataset(train_examples),
            val=ClassificationDataset(val_examples),
            test=ClassificationDataset(test_examples),
        )
    else:
        raise ValueError(f"Unsupported LRA task: {task}")

    train_tokenized = _tokenize_rows(
        train_rows,
        tokenizer,
        max_length=resolved_max_length,
        append_bos=spec.append_bos,
        append_eos=spec.append_eos,
    )
    vocab = build_vocab(
        (tokens for tokens, _ in train_tokenized),
        min_freq=spec.min_freq,
        append_bos=spec.append_bos,
        append_eos=spec.append_eos,
    )
    val_tokenized = _tokenize_rows(
        val_rows,
        tokenizer,
        max_length=resolved_max_length,
        append_bos=spec.append_bos,
        append_eos=spec.append_eos,
    )
    test_tokenized = _tokenize_rows(
        test_rows,
        tokenizer,
        max_length=resolved_max_length,
        append_bos=spec.append_bos,
        append_eos=spec.append_eos,
    )

    return LRADatasets(
        spec=spec,
        vocab=vocab,
        pad_value=vocab.pad_id,
        train=ClassificationDataset(
            _materialize_token_examples(train_tokenized, vocab, append_bos=spec.append_bos, append_eos=spec.append_eos)
        ),
        val=ClassificationDataset(
            _materialize_token_examples(val_tokenized, vocab, append_bos=spec.append_bos, append_eos=spec.append_eos)
        ),
        test=ClassificationDataset(
            _materialize_token_examples(test_tokenized, vocab, append_bos=spec.append_bos, append_eos=spec.append_eos)
        ),
    )
