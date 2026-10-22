# Bulletin Search: Item-Level Retrieval over a Newsletter Archive in Rust

A conservation studio wants to search 220 back issues of its newsletter. Each issue is a markdown file: a masthead line followed by a dozen or so short items that have little to do with each other. Given a question such as "which issue was the one about moth damage in the tapestry", the agent's Rust program (a Cargo project with `candle`, `hf-hub` and `tokenizers` already wired in) has to rank the ten most relevant issues per question and write them to `/app/rankings.json`. The agent gets 40 dev questions with answers and 60 graded questions without them. The `sentence-transformers/all-MiniLM-L6-v2` checkpoint is offered at a pinned revision, but how the program ranks is up to the agent. It must also work on a later offline run against new questions and renumbered issue files.

## What makes it hard

The deciding choice is the unit of retrieval. It is easy to get wrong, and nothing signals the mistake until the score comes back.

An issue is about 700 tokens holding a dozen unrelated items, and a question is about exactly one of them. The checkpoint is a mean-pooling encoder, so embedding a whole issue averages the relevant item with eleven others. Lexical scoring over the whole issue gets diluted the same way. Common chunk sizes are much larger than an item, so ordinary chunking doesn't help either. Measured mean nDCG@10 (graded / held-back) from `solution/calibrate.py`:

| Route | Graded | Held-back |
|---|---|---|
| Embed per item, score issue by best item (reference) | 0.858 | 0.826 |
| BM25 over items, no model | 0.872 | 0.922 |
| 126-token sliding window, 50% overlap | 0.468 | 0.438 |
| BM25 over whole issues | 0.358 | 0.459 |
| Fixed 254-token windows | 0.257 | 0.211 |
| Whole issue, truncated at 256 tokens | 0.147 | 0.224 |

Item-level BM25 clears the bars, so the task deliberately does not require the model. What it grades is granularity. Two further things also have to hold:

- Pooling must use the attention mask, or an item's vector depends on what else was in its batch.
- The program must read whatever query file and corpus it is handed, and use the cached checkpoint without network access at run time.

## How it's graded

The verifier runs in a separate container with no network and the checkpoint pre-baked into its hf-hub cache. It collects only the agent's crate sources. The corpus and queries come from the verifier's own copies. The crate is rebuilt with `--offline` as an unprivileged user over a pre-compiled dependency tree, then run twice:

1. **Graded run** on the 60 shipped questions. Mean nDCG@10 must be at least 0.65.
2. **Held-back run** on 60 new questions over the same issues with renumbered ids (`tests/remap_corpus.py` shifts every file number, rewrites mastheads, and remaps the answer key). Mean nDCG@10 must be at least 0.60. A program that memorised ids or the visible questions fails here.

Five checks in total: `Cargo.toml` and `Cargo.lock` byte-identical to the shipped versions; well-formed output on both runs (one entry per question, exactly ten distinct valid issue ids); and the two nDCG bars. Agent code runs as `runner` with `/tests` closed to it. Its process group is torn down after each run, outputs are staged to root-owned files (non-regular files are discarded), and pytest runs with `-P`. Documented arms: a whole-issue embedding variant scores 0 (0.089 / 0.194), and a program with the graded answers compiled in scores 0 on the held-back run.

## Repository layout

- `instruction.md` - the task prompt given to the agent
- `task.toml` - metadata, limits, and the difficulty, solution and verification write-ups
- `environment/` - agent container: `Dockerfile`, `corpus/` (220 issues), `queries/` (dev and graded questions), `search/` (Cargo project with stub `main.rs`)
- `solution/` - `solve.sh`, reference `main.rs`, `embed.py`, `generate_corpus.py` and `corpus_lib.py` (regenerate the fixtures from a seed), `calibrate.py` (re-derives the route table)
- `tests/` - verifier `Dockerfile`, `test.sh`, `test_outputs.py`, `remap_corpus.py`, `warmup.rs`, pre-built `skeleton/`, the verifier's `corpus/` and `queries/`, `golden/` and `hidden/` answer keys

## Running it

Requires Docker and Harbor (`uv tool install harbor` or `pip install harbor`). From the repo root:

```
harbor run -p . -a oracle
```

runs the reference solution through the verifier (should score reward 1). To try an agent:

```
harbor run -p . -a <agent> -m <model>
```

for example `-a claude-code`.

---

Task design, data generators, reference solution and verifier by Audrey Pope.
