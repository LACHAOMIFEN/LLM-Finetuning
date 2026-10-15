"""Build reproducible, disjoint instruction tuning splits from official TNEWS."""
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LABELS = {
    "100": "故事", "101": "文化", "102": "娱乐", "103": "体育",
    "104": "财经", "106": "房产", "107": "汽车", "108": "教育",
    "109": "科技", "110": "军事", "112": "旅游", "113": "国际",
    "114": "股票", "115": "农业", "116": "游戏",
}
SYSTEM_PROMPT = (
    "你是中文新闻分类助手。请根据新闻标题，从以下类别中选择一个最合适的类别："
    + "、".join(LABELS.values())
    + '。只输出一个JSON对象，格式为{"category":"类别"}，不要解释。'
)


def normalized(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sample_stratified(rows, count, seed):
    """Sample approximately in proportion to class frequency, with exact size."""
    if not 0 < count <= len(rows):
        raise ValueError((count, len(rows)))
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    quotas = {key: count * len(group) / len(rows) for key, group in groups.items()}
    sizes = {key: int(value) for key, value in quotas.items()}
    for key in sorted(groups, key=lambda k: (-(quotas[k] - sizes[k]), k))[:count - sum(sizes.values())]:
        sizes[key] += 1
    selected, remaining = [], []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        selected.extend(group[:sizes[key]])
        remaining.extend(group[sizes[key]:])
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def messages(title):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "新闻标题：" + title}]


def main():
    raw, dest = Path("data/raw"), Path("data/tnews")
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise RuntimeError("data/tnews already contains data; keep existing splits or use a new directory")
    seen, clean, dropped = set(), {}, {}
    # Give the official development split priority; remove overlaps from train.
    for split in ["dev", "train"]:
        clean[split] = []
        rows = read_jsonl(raw / f"{split}.json")
        for index, row in enumerate(rows):
            key = normalized(row["sentence"])
            if key and key not in seen:
                assert row["label"] in LABELS
                seen.add(key)
                clean[split].append({**row, "source_split": split, "source_row": index})
        dropped[split] = len(rows) - len(clean[split])
    train, remaining = sample_stratified(clean["train"], 2000, 42)
    val, _ = sample_stratified(remaining, 200, 43)
    test, _ = sample_stratified(clean["dev"], 300, 44)
    splits = {"train": train, "validation": val, "test": test}
    title_sets = [set(normalized(r["sentence"]) for r in rows) for rows in splits.values()]
    assert all(not title_sets[i] & title_sets[j] for i in range(3) for j in range(i))
    counts = {}
    for split, rows in splits.items():
        formatted = []
        for row in rows:
            category = LABELS[row["label"]]
            answer = json.dumps({"category": category}, ensure_ascii=False, separators=(",", ":"))
            formatted.append({
                "id": f'{row["source_split"]}:{row["source_row"]}',
                "title": row["sentence"], "label": row["label"], "category": category,
                "messages": messages(row["sentence"]) + [{"role": "assistant", "content": answer}],
            })
        (dest / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in formatted))
        counts[split] = dict(Counter(row["category"] for row in formatted))
    manifest = {
        "dataset": "CLUE TNEWS", "url": "https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip",
        "archive_sha256": hashlib.sha256((raw / "tnews_public.zip").read_bytes()).hexdigest(),
        "labels": LABELS, "counts": counts, "removed_empty_or_duplicate_titles": dropped,
        "split_seeds": {"train": 42, "validation": 43, "test": 44},
        "test_note": "Local held-out subset of official labeled dev split; not official leaderboard test.",
        "deduplication": "NFKC and whitespace normalization; exact title duplicates only; dev has priority.",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
