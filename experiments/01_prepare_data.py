#!/usr/bin/env python3
"""
01_prepare_data.py — Parse filenames, deduplicate, assign categories, stratified split.

Output:
  data/metadata.csv       — full metadata (unique paintings only)
  data/train.csv          — 80% stratified
  data/val.csv            — 10%
  data/test.csv           — 10%
  data/class_weights.json — inverse-frequency weights for imbalanced training
  data/stats.json         — summary statistics for paper
  data/dedup_log.json     — deduplication decisions log
"""
import os, json, re, sys
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_ROOT, DATA_DIR, SEED, SPLIT_RATIO,
    TECHNIQUE_MAP, SUBJECT_MAP, MERGE_RULES, MERGE_THRESHOLD,
    EXCLUDE_KEYWORDS, EXCLUDE_PREFIX,
)

np.random.seed(SEED)


def parse_filename(fname: str) -> dict | None:
    """Parse VULCA 2.0 Chinese painting filename into metadata dict."""
    if not fname.endswith(".jpg"):
        return None

    # Exclude by prefix
    if fname.startswith(EXCLUDE_PREFIX):
        return None

    # Exclude by keywords
    for kw in EXCLUDE_KEYWORDS:
        if kw in fname:
            return None

    # Filename pattern: _朝代_作者_作品名_形式_技法,分类_元素_材质_尺寸_来源_hash.jpg
    # Fields are underscore-separated, but many have spaces and missing values
    # Key: find the 技法,分类 field (comma-separated pair)
    base = fname.rsplit(".", 1)[0]  # remove .jpg
    parts = base.split("_")

    technique_zh = None
    subject_zh = None

    for p in parts:
        p_stripped = p.strip()
        if "," in p_stripped:
            comma_parts = p_stripped.split(",")
            if len(comma_parts) >= 2:
                t_candidate = comma_parts[0].strip()
                s_candidate = comma_parts[1].strip()
                if t_candidate in TECHNIQUE_MAP and s_candidate in SUBJECT_MAP:
                    technique_zh = t_candidate
                    subject_zh = s_candidate
                    break
                # Sometimes subject is in a different position
                elif t_candidate in TECHNIQUE_MAP:
                    technique_zh = t_candidate
                    # Look for subject in remaining comma parts
                    for cp in comma_parts[1:]:
                        if cp.strip() in SUBJECT_MAP:
                            subject_zh = cp.strip()
                            break

    if technique_zh is None or subject_zh is None:
        return None

    # Extract other fields (best-effort)
    # Try to get dynasty and author from first parts
    dynasty = parts[0].strip() if len(parts) > 0 and parts[0].strip() else None
    author = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    title = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    form = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None

    return {
        "filename": fname,
        "dynasty": dynasty,
        "author": author,
        "title": title,
        "form": form,
        "technique_zh": technique_zh,
        "subject_zh": subject_zh,
        "technique": TECHNIQUE_MAP[technique_zh],
        "subject": SUBJECT_MAP[subject_zh],
    }


HASH_RE = re.compile(r'_([0-9a-f]{8})\.jpg$')


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Remove duplicate paintings (same content hash, different resolutions).

    For each group sharing an 8-char hex hash, keep the file with the
    largest file size (highest resolution).  Returns (deduped_df, log).
    """
    df = df.copy()
    # Extract content hash from filename
    df["_hash"] = df["filename"].apply(
        lambda f: m.group(1) if (m := HASH_RE.search(f)) else None
    )

    # Files without hash are unique by definition
    no_hash = df[df["_hash"].isna()]
    has_hash = df[df["_hash"].notna()]

    # Group by hash, keep highest-resolution version
    dedup_log = []
    keep_idx = []

    for h, group in has_hash.groupby("_hash"):
        if len(group) == 1:
            keep_idx.append(group.index[0])
        else:
            # Compare file sizes to pick highest resolution
            sizes = {}
            for idx, row in group.iterrows():
                fpath = os.path.join(DATA_ROOT, row["filename"])
                sizes[idx] = os.path.getsize(fpath) if os.path.exists(fpath) else 0

            best_idx = max(sizes, key=sizes.get)
            keep_idx.append(best_idx)

            removed = [i for i in group.index if i != best_idx]
            dedup_log.append({
                "hash": h,
                "kept": df.loc[best_idx, "filename"],
                "kept_size": sizes[best_idx],
                "removed": [
                    {"filename": df.loc[i, "filename"], "size": sizes[i]}
                    for i in removed
                ],
            })

    result = pd.concat([no_hash, has_hash.loc[keep_idx]]).sort_index()
    result = result.drop(columns=["_hash"])
    return result, dedup_log


def assign_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Assign 3-class and fine-grained categories with merging."""
    # 3-class: just the subject
    df["category_3"] = df["subject"]

    # Fine-grained: technique × subject
    df["category_raw"] = df["technique"] + "-" + df["subject"]

    # Apply merge rules for small classes
    df["category_fine"] = df["category_raw"].copy()

    # First pass: apply explicit merge rules
    for src, dst in MERGE_RULES.items():
        mask = df["category_fine"] == src
        if mask.sum() > 0:
            df.loc[mask, "category_fine"] = dst

    # Second pass: merge any remaining class < threshold
    while True:
        counts = df["category_fine"].value_counts()
        small = counts[counts < MERGE_THRESHOLD]
        if len(small) == 0:
            break

        merged_any = False
        for cls_name in small.index:
            # Find nearest larger class (same subject if possible)
            parts = cls_name.split("-")
            if len(parts) == 2:
                tech, subj = parts
                # Try same-subject classes
                candidates = counts[
                    (counts.index.str.endswith(f"-{subj}")) &
                    (counts.index != cls_name) &
                    (counts >= MERGE_THRESHOLD)
                ]
                if len(candidates) > 0:
                    target = candidates.idxmax()  # merge into largest same-subject class
                    df.loc[df["category_fine"] == cls_name, "category_fine"] = target
                    merged_any = True
                else:
                    # Merge into largest overall class with same subject
                    all_same_subj = counts[
                        (counts.index.str.endswith(f"-{subj}")) &
                        (counts.index != cls_name)
                    ]
                    if len(all_same_subj) > 0:
                        target = all_same_subj.idxmax()
                        df.loc[df["category_fine"] == cls_name, "category_fine"] = target
                        merged_any = True

        if not merged_any:
            break

    return df


def stratified_split(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/10/10 stratified split."""
    train_ratio, val_ratio, test_ratio = SPLIT_RATIO

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        stratify=df[label_col], random_state=SEED
    )

    # Second split: val vs test (50/50 of the remaining 20%)
    val_df, test_df = train_test_split(
        temp_df, test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df[label_col], random_state=SEED
    )

    return train_df, val_df, test_df


def compute_class_weights(train_df: pd.DataFrame, label_col: str) -> dict[str, float]:
    """Inverse-frequency class weights for weighted loss."""
    counts = train_df[label_col].value_counts()
    total = len(train_df)
    n_classes = len(counts)
    weights = {}
    for cls, cnt in counts.items():
        weights[cls] = total / (n_classes * cnt)
    return weights


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    # ──── Parse all filenames ────
    all_files = sorted(os.listdir(DATA_ROOT))
    print(f"Total files in directory: {len(all_files)}")

    records = []
    skipped_reasons = Counter()
    for fname in all_files:
        result = parse_filename(fname)
        if result is not None:
            records.append(result)
        else:
            # Categorize skip reason
            if fname.startswith(EXCLUDE_PREFIX):
                skipped_reasons["ch_prefix"] += 1
            elif any(kw in fname for kw in EXCLUDE_KEYWORDS):
                skipped_reasons["excluded_keyword"] += 1
            elif not fname.endswith(".jpg"):
                skipped_reasons["not_jpg"] += 1
            else:
                skipped_reasons["unparseable"] += 1

    df = pd.DataFrame(records)
    print(f"Parsed records (before dedup): {len(df)}")
    print(f"Skipped: {dict(skipped_reasons)}")

    # ──── Deduplicate (same painting, different resolutions) ────
    df, dedup_log = deduplicate(df)
    print(f"Deduplicated: removed {len(dedup_log)} duplicate pairs")
    print(f"Unique paintings: {len(df)}")

    # Save dedup log
    with open(os.path.join(DATA_DIR, "dedup_log.json"), "w") as f:
        json.dump(dedup_log, f, indent=2, ensure_ascii=False)

    # ──── Assign categories ────
    df = assign_categories(df)

    # ──── Report ────
    print(f"\n=== 3-Class Distribution ===")
    cat3 = df["category_3"].value_counts()
    for cls, cnt in cat3.items():
        print(f"  {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print(f"\n=== Fine-Grained Distribution (after merging) ===")
    catf = df["category_fine"].value_counts()
    for cls, cnt in catf.items():
        print(f"  {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")

    n_fine = len(catf)
    print(f"\nFinal: {len(df)} images, 3 coarse classes, {n_fine} fine-grained classes")

    # ──── Dynasty distribution ────
    dynasty_counts = df["dynasty"].value_counts()
    print(f"\n=== Dynasty Distribution (top 10) ===")
    for d, cnt in dynasty_counts.head(10).items():
        print(f"  {d}: {cnt}")

    # ──── Stratified split on fine-grained ────
    train_df, val_df, test_df = stratified_split(df, "category_fine")

    print(f"\n=== Split ===")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Verify no overlap (filename level)
    train_files = set(train_df["filename"])
    val_files = set(val_df["filename"])
    test_files = set(test_df["filename"])
    assert len(train_files & val_files) == 0, "Train-val overlap!"
    assert len(train_files & test_files) == 0, "Train-test overlap!"
    assert len(val_files & test_files) == 0, "Val-test overlap!"
    print("  No filename overlap ✓")

    # Verify no content-level overlap (hash level)
    def get_hashes(split_df):
        hashes = set()
        for f in split_df["filename"]:
            m = HASH_RE.search(f)
            if m:
                hashes.add(m.group(1))
        return hashes

    train_h = get_hashes(train_df)
    val_h = get_hashes(val_df)
    test_h = get_hashes(test_df)
    assert len(train_h & val_h) == 0, f"Train-val hash overlap: {len(train_h & val_h)}!"
    assert len(train_h & test_h) == 0, f"Train-test hash overlap: {len(train_h & test_h)}!"
    assert len(val_h & test_h) == 0, f"Val-test hash overlap: {len(val_h & test_h)}!"
    print("  No content-hash overlap ✓")

    # Verify each class represented in all splits
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        classes = set(split_df["category_fine"].unique())
        all_classes = set(df["category_fine"].unique())
        missing = all_classes - classes
        if missing:
            print(f"  WARNING: {split_name} missing classes: {missing}")
        else:
            print(f"  {split_name}: all {len(all_classes)} classes represented ✓")

    # ──── Save ────
    df.to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    # Class weights for both tasks
    weights_3 = compute_class_weights(train_df, "category_3")
    weights_fine = compute_class_weights(train_df, "category_fine")
    weights = {"category_3": weights_3, "category_fine": weights_fine}
    with open(os.path.join(DATA_DIR, "class_weights.json"), "w") as f:
        json.dump(weights, f, indent=2)

    # Summary stats for paper
    stats = {
        "total_files": len(all_files),
        "parsed_before_dedup": len(df) + len(dedup_log),
        "duplicates_removed": len(dedup_log),
        "unique_paintings": len(df),
        "skipped": dict(skipped_reasons),
        "n_classes_3": 3,
        "n_classes_fine": n_fine,
        "class_3_distribution": {cls: int(cnt) for cls, cnt in cat3.items()},
        "class_3_percentages": {cls: round(cnt/len(df)*100, 1) for cls, cnt in cat3.items()},
        "class_fine_distribution": {cls: int(cnt) for cls, cnt in catf.items()},
        "split_train": len(train_df),
        "split_val": len(val_df),
        "split_test": len(test_df),
        "dynasty_distribution": {str(d): int(cnt) for d, cnt in dynasty_counts.head(15).items()},
    }
    with open(os.path.join(DATA_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nAll outputs saved to {DATA_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
