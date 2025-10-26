import csv, json, random
from pathlib import Path
from typing import List, Tuple, Dict

def read_text_label_csv(path: str) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            texts.append((row.get("text") or "").strip())
            labels.append(int(row.get("label")))
    return texts, labels

def ensure_split_indices(n_items: int, split_path: str, seed: int = 340) -> Dict[str, list]:
    p = Path(split_path)
    if p.exists():
        return json.loads(p.read_text())
    idx = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_train = int(0.8 * n_items)
    n_valid = int(0.1 * n_items)
    split = {
        "seed": seed,
        "train_idx": idx[:n_train],
        "valid_idx": idx[n_train:n_train + n_valid],
        "test_idx": idx[n_train + n_valid:],
    }
    p.write_text(json.dumps(split))
    return split

def take_by_index(xs: List, ids: List[int]) -> List:
    return [xs[i] for i in ids]

def make_splits(texts: List[str], labels: List[int], split: Dict[str, list]):
    tr = split["train_idx"]; va = split["valid_idx"]; te = split["test_idx"]
    Xtr = take_by_index(texts, tr); Ytr = take_by_index(labels, tr)
    Xva = take_by_index(texts, va); Yva = take_by_index(labels, va)
    Xte = take_by_index(texts, te); Yte = take_by_index(labels, te)
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)
