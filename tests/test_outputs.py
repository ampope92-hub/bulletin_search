"""Grades the rankings two runs of the agent's program produced.

The graded run answers the query file that shipped in the agent's environment; the hidden
run answers a file it has never seen, over the same corpus. Both are read from root-owned
captures staged by test.sh, never from a path the agent's code can still write.
"""
import json
import math
import os
import pathlib

import pytest

GRADED_RUN = pathlib.Path("/opt/run_graded.json")
HIDDEN_RUN = pathlib.Path("/opt/run_hidden.json")
GRADED_LABELS = pathlib.Path("/opt/golden/graded_labels.json")
# The held-back run is over the same issues under fresh ids (test.sh renames every file by a
# fixed shift and carries the answer key through the same map), so its output is checked
# against the corpus it was actually handed and the labels as remapped.
HIDDEN_LABELS = pathlib.Path("/opt/hidden/labels_remapped.json")
CORPUS = pathlib.Path("/opt/corpus")
CORPUS_HIDDEN = pathlib.Path("/opt/corpus_hidden")

TOP_K = 10
# Calibrated in solution/calibrate.py against the shipped fixtures. Splitting each issue on
# its own item boundaries scores 0.858 on the graded queries and 0.826 on the held-back ones
# with the MiniLM checkpoint; BM25 over the same item split, with no model at all, scores
# 0.872 and 0.922. Nothing that stops short of item granularity gets close with either: a
# 126-token sliding window reaches 0.468 and 0.438, whole-issue embeddings 0.147 and 0.224,
# and BM25 over whole issues 0.358 and 0.459. So the axis this task grades is granularity,
# not the choice of model, and the bars sit under both item-level routes with margin.
GRADED_BAR = 0.65
# Matches the figure instruction.md states for the held-back run, and is deliberately
# slacker than the graded bar so nothing enforced here is tighter than what the agent was
# told. The reference scores 0.826 on this run, so the margin is not doing any work.
HIDDEN_BAR = 0.60


def _read_json(path, what):
    if not path.is_file():
        pytest.fail(f"{what} is missing: {path} was never produced")
    with open(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{what} is not valid JSON: {exc}")


def _issue_ids(corpus):
    return {p.stem for p in corpus.glob("issue_*.md")}


def _as_map(rows, what):
    if not isinstance(rows, list):
        pytest.fail(f"{what} must be a JSON array of objects, got {type(rows).__name__}")
    out = {}
    for row in rows:
        if not isinstance(row, dict) or "query_id" not in row or "issues" not in row:
            pytest.fail(f"{what} has an entry without query_id and issues: {row!r}")
        if row["query_id"] in out:
            pytest.fail(f"{what} lists query {row['query_id']} more than once")
        out[row["query_id"]] = row["issues"]
    return out


def _check_shape(rows, labels, what, corpus=CORPUS):
    ranked = _as_map(rows, what)
    valid = _issue_ids(corpus)
    missing = sorted(set(labels) - set(ranked))
    assert not missing, f"{what} has no ranking for {len(missing)} queries, e.g. {missing[:5]}"
    extra = sorted(set(ranked) - set(labels))
    assert not extra, f"{what} ranks queries that were never asked: {extra[:5]}"
    for qid, issues in ranked.items():
        assert isinstance(issues, list), f"{what}: {qid} issues is not a list"
        assert len(issues) == TOP_K, (
            f"{what}: {qid} lists {len(issues)} issues, expected exactly {TOP_K}")
        assert len(set(issues)) == TOP_K, (
            f"{what}: {qid} lists an issue more than once; the ten ids have to be different")
        unknown = [i for i in issues if i not in valid]
        assert not unknown, f"{what}: {qid} names issues not in the corpus: {unknown[:3]}"
    return ranked


def _ndcg(ranked, labels):
    total = 0.0
    for qid, relevant in labels.items():
        rel = set(relevant)
        got = ranked[qid][:TOP_K]
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(got) if d in rel)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(TOP_K, len(rel))))
        total += dcg / ideal if ideal else 0.0
    return total / len(labels)


def test_graded_rankings_are_well_formed():
    """Every graded question is answered with ten distinct issue ids from the corpus."""
    labels = _read_json(GRADED_LABELS, "graded answer key")
    _check_shape(_read_json(GRADED_RUN, "graded run output"), labels, "graded run output")


def test_graded_ndcg_clears_the_bar():
    """Mean nDCG@10 on the sixty graded questions reaches the 0.65 the instruction states."""
    labels = _read_json(GRADED_LABELS, "graded answer key")
    ranked = _check_shape(_read_json(GRADED_RUN, "graded run output"), labels,
                          "graded run output")
    score = _ndcg(ranked, labels)
    assert score >= GRADED_BAR, (
        f"nDCG@10 on the graded queries is {score:.4f}, below the required {GRADED_BAR}")


def test_hidden_rankings_are_well_formed():
    """The held-back run answers every question it was handed, in the same shape."""
    labels = _read_json(HIDDEN_LABELS, "hidden answer key")
    _check_shape(_read_json(HIDDEN_RUN, "hidden run output"), labels, "hidden run output",
                 CORPUS_HIDDEN)


def test_hidden_ndcg_clears_the_bar():
    """Mean nDCG@10 on questions the program never saw reaches the 0.60 bar the instruction
    names for that run, over the corpus as it was handed to that run."""
    labels = _read_json(HIDDEN_LABELS, "hidden answer key")
    ranked = _check_shape(_read_json(HIDDEN_RUN, "hidden run output"), labels,
                          "hidden run output", CORPUS_HIDDEN)
    score = _ndcg(ranked, labels)
    assert score >= HIDDEN_BAR, (
        f"nDCG@10 on the held-back queries is {score:.4f}, below the required {HIDDEN_BAR}. "
        "The program answered the queries it shipped with but not queries it had not seen, or "
        "it answered from a copy of the corpus rather than the one it was pointed at")


def test_manifest_and_lockfile_are_as_shipped():
    """Cargo.toml and Cargo.lock come back byte for byte as issued. test.sh compares the collected
    copies with the shipped ones before building and always builds with the shipped ones, so a
    crate promoted to a direct dependency from the cache cannot ride in on a regenerated lock."""
    marker = pathlib.Path("/opt/manifest_check.txt")
    assert marker.is_file() and marker.read_text().strip() == "ok", (
        "Cargo.toml or Cargo.lock was changed; the instruction says to stay with the crates "
        "already listed and leave Cargo.lock alone")
