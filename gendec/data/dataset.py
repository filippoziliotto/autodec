from pathlib import Path

from torch.utils.data import Dataset

from gendec.data.examples import load_teacher_example
from gendec.data.layout import iter_exported_examples, normalization_stats_path
from gendec.data.normalization import load_normalization_stats, normalize_tokens


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
        return item


load_normalization_stats = load_normalization_stats
