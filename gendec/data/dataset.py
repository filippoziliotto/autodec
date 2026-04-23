from pathlib import Path

from torch.utils.data import Dataset
import torch

from gendec.data.examples import load_teacher_example
from gendec.data.layout import build_category_vocab, iter_exported_examples, normalization_stats_path
from gendec.data.normalization import load_normalization_stats, normalize_tokens
from gendec.tokens import TOKEN_DIM


class ScaffoldTokenDataset(Dataset):
    def __init__(self, root, split=None, categories=None):
        self.root = Path(root)
        self.split = split
        self.categories = categories
        self.index = list(
            iter_exported_examples(
                self.root,
                split=self.split,
                categories=self.categories,
            )
        )
        if not self.index:
            raise FileNotFoundError(f"No scaffold examples found in {self.root}")
        self.category_ids, self.category_to_index = build_category_vocab(self.root, categories=self.categories)
        self.num_classes = len(self.category_ids)
        self.files = [item["path"] for item in self.index]
        self.models = [
            {"category_id": item["category_id"], "model_id": item["model_id"]}
            for item in self.index
        ]
        self.stats = load_normalization_stats(normalization_stats_path(self.root))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        example = load_teacher_example(self.index[idx]["path"])
        tokens_e_raw = example["tokens_e"].to(dtype=self.stats["mean"].dtype)
        item = dict(example)
        item["tokens_e_raw"] = tokens_e_raw
        item["tokens_e"] = normalize_tokens(tokens_e_raw, self.stats)
        item["token_mean"] = self.stats["mean"].clone()
        item["token_std"] = self.stats["std"].clone()
        item["category_index"] = torch.tensor(self.category_to_index[str(example["category_id"])], dtype=torch.long)
        return item


class JointTokenDataset(Dataset):
    """Phase 2 dataset that loads joint (E, Z) tokens from teacher examples.

    The on-disk examples must contain ``tokens_ez`` (written by
    ``build_toy_phase2_example`` or the real Phase 2 export). Normalization
    statistics are computed/loaded over the full 79D joint token.

    Each sample returned contains:
        tokens_ez       [16, 79]  – normalized
        tokens_ez_raw   [16, 79]  – unnormalized
        tokens_e        [16, 15]  – explicit scaffold slice (unnormalized)
        tokens_z        [16, D]   – residual slice (unnormalized)
        exist           [16, 1]
        mass            [16]
        volume          [16]
        token_mean      [79]
        token_std       [79]
        category_id     str
        model_id        str
    """

    def __init__(self, root, split=None, categories=None):
        self.root = Path(root)
        self.split = split
        self.categories = categories
        self.index = list(
            iter_exported_examples(
                self.root,
                split=self.split,
                categories=self.categories,
            )
        )
        if not self.index:
            raise FileNotFoundError(f"No Phase 2 joint examples found in {self.root}")
        self.category_ids, self.category_to_index = build_category_vocab(self.root, categories=self.categories)
        self.num_classes = len(self.category_ids)
        self.files = [item["path"] for item in self.index]
        self.models = [
            {"category_id": item["category_id"], "model_id": item["model_id"]}
            for item in self.index
        ]
        self.stats = load_normalization_stats(normalization_stats_path(self.root))

    @property
    def residual_dim(self):
        return int(self.stats["mean"].shape[0]) - TOKEN_DIM

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        example = load_teacher_example(self.index[idx]["path"])
        if "tokens_ez" not in example:
            raise KeyError(
                f"Example at {self.index[idx]['path']} has no 'tokens_ez' key. "
                "Use write_toy_phase2_dataset or the Phase 2 real export."
            )
        tokens_ez_raw = example["tokens_ez"].to(dtype=self.stats["mean"].dtype)
        item = dict(example)
        item["tokens_ez_raw"] = tokens_ez_raw
        item["tokens_ez"] = normalize_tokens(tokens_ez_raw, self.stats)
        item["tokens_e"] = tokens_ez_raw[..., :TOKEN_DIM]
        item["tokens_z"] = tokens_ez_raw[..., TOKEN_DIM:]
        item["token_mean"] = self.stats["mean"].clone()
        item["token_std"] = self.stats["std"].clone()
        item["category_index"] = torch.tensor(self.category_to_index[str(example["category_id"])], dtype=torch.long)
        return item


load_normalization_stats = load_normalization_stats
