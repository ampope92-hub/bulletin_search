#!/usr/bin/env python3
"""Write every fixture this task ships: the corpus, the three query files and the two
answer keys. Deterministic from SEED -- rerunning it reproduces the files byte for byte.

  corpus            environment/corpus/issue_0001.md .. issue_0220.md
  labelled examples environment/queries/dev.json        (40, with issue ids)
  graded            environment/queries/graded.json     (60, no issue ids)
  hidden            tests/hidden/queries.json           (60, no issue ids)
  answer keys       tests/golden/graded_labels.json, tests/hidden/labels.json

The hidden set is drawn from the same shuffle as the other two, so it is the same
distribution of queries over the same corpus -- it exists to show the program answers
queries it was never shown, not to grade the model a second time.
"""
import json
import random
import sys
from pathlib import Path

import corpus_lib

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_DOCS, ITEMS_PER_DOC, N_ITEMS = 220, 12, 1400
N_DEV, N_GRADED, N_HIDDEN = 40, 60, 60
PARAPHRASE_SLOTS = 1

root = Path(__file__).resolve().parent.parent
corpus_dir = root / "environment" / "corpus"
query_dir = root / "environment" / "queries"
hidden_dir = root / "tests" / "hidden"
golden_dir = root / "tests" / "golden"


def issue_id(d):
    return f"issue_{d + 1:04d}"


def main():
    docs, items, where = corpus_lib.build(SEED, N_DOCS, ITEMS_PER_DOC, N_ITEMS)

    for d in (corpus_dir, query_dir, hidden_dir, golden_dir):
        d.mkdir(parents=True, exist_ok=True)
    for old in corpus_dir.glob("issue_*.md"):
        old.unlink()
    for d, text in enumerate(docs):
        (corpus_dir / f"{issue_id(d)}.md").write_text(text + "\n", encoding="utf-8")

    rng = random.Random(SEED + 500)
    keys = sorted(where)
    rng.shuffle(keys)
    picked = keys[: N_DEV + N_GRADED + N_HIDDEN]

    rows = []
    for n, item_idx in enumerate(picked):
        rows.append({
            "query_id": f"q{n + 1:04d}",
            "query": corpus_lib.render_query(items[item_idx], PARAPHRASE_SLOTS, rng),
            "issues": sorted(issue_id(d) for d, _ in where[item_idx]),
        })

    dev = rows[:N_DEV]
    graded = rows[N_DEV:N_DEV + N_GRADED]
    hidden = rows[N_DEV + N_GRADED:]

    def strip(rs):
        return [{"query_id": r["query_id"], "query": r["query"]} for r in rs]

    def key(rs):
        return {r["query_id"]: r["issues"] for r in rs}

    (query_dir / "dev.json").write_text(json.dumps(dev, indent=2) + "\n")
    (query_dir / "graded.json").write_text(json.dumps(strip(graded), indent=2) + "\n")
    (hidden_dir / "queries.json").write_text(json.dumps(strip(hidden), indent=2) + "\n")
    (golden_dir / "graded_labels.json").write_text(json.dumps(key(graded), indent=2) + "\n")
    (hidden_dir / "labels.json").write_text(json.dumps(key(hidden), indent=2) + "\n")

    rel = [len(r["issues"]) for r in rows]
    print(f"seed={SEED} issues={len(docs)} items={len(items)}")
    print(f"queries dev={len(dev)} graded={len(graded)} hidden={len(hidden)} "
          f"relevant/query min={min(rel)} max={max(rel)}")


if __name__ == "__main__":
    main()
