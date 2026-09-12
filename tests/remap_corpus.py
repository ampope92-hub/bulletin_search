"""Rebuild the corpus under fresh issue ids for the held-back run.

Every issue file is renamed by a fixed cyclic shift of its number and the masthead line is
rewritten to match, and the held-back answer key is carried through the same map. An honest
program reads /app/corpus when it runs and never notices. A program that compiled the corpus
into its source, embeddings and ids alike, answers with the ids the issues used to have, which
now belong to other issues, and its held-back nDCG collapses.

    python3 remap_corpus.py <src corpus> <dst corpus> <labels in> <labels out> <id map out>
"""

import json
import pathlib
import re
import sys

SHIFT = 97


def shifted(issue_id, count):
    n = int(issue_id.split("_")[1])
    return f"issue_{((n - 1 + SHIFT) % count) + 1:04d}"


def main():
    src, dst, labels_in, labels_out, map_out = (pathlib.Path(a) for a in sys.argv[1:6])
    files = sorted(src.glob("issue_*.md"))
    count = len(files)
    dst.mkdir(parents=True, exist_ok=True)
    id_map = {}
    for path in files:
        new_id = shifted(path.stem, count)
        id_map[path.stem] = new_id
        text = path.read_text(encoding="utf-8")
        head, sep, rest = text.partition("\n")
        head = re.sub(r"issue \d+", f"issue {int(new_id.split('_')[1])}", head, count=1)
        (dst / f"{new_id}.md").write_text(head + sep + rest, encoding="utf-8")
    labels = json.loads(labels_in.read_text(encoding="utf-8"))
    labels_out.write_text(
        json.dumps({q: [id_map[i] for i in ids] for q, ids in labels.items()}, indent=2),
        encoding="utf-8",
    )
    map_out.write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    print(f"remapped {count} issues by a shift of {SHIFT}")


if __name__ == "__main__":
    main()
