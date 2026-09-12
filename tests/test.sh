#!/bin/bash

# Verifier entrypoint. Runs in the verifier container, not the agent's, and sees only the
# paths named in `artifacts`. All tooling is baked into tests/Dockerfile; nothing is fetched
# here -- [verifier].network_mode is "no-network" and the checkpoint is already on disk.

set -uo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

# Fold stderr into stdout. If this script dies between the two runs, the reason belongs in
# the same log as everything else rather than on a stream nobody keeps.
exec 2>&1


BUILD=/opt/skeleton

# The rebuild executes code the agent wrote. As root that code could read the answer keys and
# echo them back, or rewrite test_outputs.py before pytest reads it, so it runs as `runner`
# and the verifier's own files are closed first.
chmod 700 /tests
echo "verifier files closed to the unprivileged build user"
as_runner() {
  setpriv --clear-groups --no-new-privs --reuid runner --regid runner \
    env HOME=/home/runner HF_HOME=/home/runner/.cache/huggingface "$@"
}

# Snapshot what the agent left behind before anything in here overwrites it.

# Assemble the workspace from this image's own copies. The corpus and the queries are never
# collected from the agent, so both runs below are provably over inputs it could not touch.
rm -rf /app/corpus /app/queries /app/rankings.json
cp -a /opt/corpus /app/corpus
mkdir -p /app/queries
cp /opt/queries/graded.json /app/queries/graded.json
chown runner:runner /app

# Build the agent's sources against the dependency tree already compiled into this image,
# using the manifest and lockfile that shipped, never the agent's copies. The instruction says
# to stay with the crates in Cargo.toml and leave Cargo.lock alone, so the two collected files
# are compared byte for byte with the originals and the result is recorded for pytest. A crate
# already in the cache through another dependency could otherwise be promoted to a direct one
# and still build offline, which --locked on the agent's own lockfile would never notice.
if cmp -s /app/search/Cargo.toml "$BUILD/Cargo.toml" && cmp -s /app/search/Cargo.lock "$BUILD/Cargo.lock"; then
  echo ok > /opt/manifest_check.txt
else
  echo changed > /opt/manifest_check.txt
  echo "Cargo.toml or Cargo.lock differs from the shipped copy"
fi
chown root:root /opt/manifest_check.txt; chmod 600 /opt/manifest_check.txt
rm -rf "$BUILD/src"
cp -a /app/search/src "$BUILD/src" 2>/dev/null || true
# Cargo decides freshness by mtime, and collected sources can predate this image's layers, so
# an older file would silently count as already built. Stamp them and drop the crate's own
# object files. Belt and braces with the `cargo clean -p` in tests/Dockerfile.
find "$BUILD/src" -type f -exec touch {} + 2>/dev/null || true
chown -R runner:runner "$BUILD"
as_runner bash -c "cd $BUILD && cargo clean -p bulletin_search --release" 2>/dev/null || true
as_runner bash -c "cd $BUILD && cargo build --release --locked --offline"
echo "build exit status: $?"

# Run the program in its own session so everything it spawns shares one process group that
# can be torn down as a unit. A double-forked child left alive would still be around during
# the second run, where it could write /app/rankings.json and pass the first run's answers
# off as the second's. [verifier].timeout_sec is the only bound on either run: a wall-clock
# wrapper here can fail a slow but correct program.
run_once() {
  local label="$1" capture="$2"
  rm -f /app/rankings.json
  # setsid has to come first and be given a real executable: it execs its argument, so a
  # shell function cannot be handed to it.
  setsid setpriv --clear-groups --no-new-privs --reuid runner --regid runner \
    env HOME=/home/runner HF_HOME=/home/runner/.cache/huggingface \
    bash -c "cd $BUILD && ./target/release/bulletin_search" &
  local pid=$!
  # /proc rather than ps: procps is not in this image. Fields after "comm)" are state, ppid, pgrp.
  wait "$pid"
  echo "$label exit status: $?"
  # setsid makes the run a session leader, and a session leader's process group id is its own
  # pid, so the group is addressed directly. Reading the group back out of /proc instead races
  # setsid(2): in the window before it takes effect /proc still reports the child in THIS
  # shell's group, and signalling that group kills the verifier itself, silently, with no
  # pytest output and the reward left at 0.
  kill -TERM -- -"$pid" 2>/dev/null || true
  kill -KILL -- -"$pid" 2>/dev/null || true
  # Written by runner, so the move keeps runner's ownership. Hand it to root before any more
  # agent code runs, or the second run could rewrite the first run's captured output. A
  # symlink planted at that path would be followed when pytest reads it as root, so anything
  # that is not a regular file is discarded instead of staged.
  if [ -L /app/rankings.json ] || [ ! -f /app/rankings.json ]; then
    echo "$label produced no regular rankings.json"
    return 0
  fi
  mv /app/rankings.json "$capture"
  chown root:root "$capture" 2>/dev/null || true
  chmod 600 "$capture" 2>/dev/null || true
}

run_once graded /opt/run_graded.json

# Second run, on queries the agent has never seen, over the same issues under fresh ids: every
# file is renamed by a fixed shift of its number, its masthead rewritten to match, and the
# answer key carried through the same map. A program that reads /app/corpus when it runs never
# notices. One that answered the shipped queries by any means other than searching the corpus
# it was handed, including one that compiled the corpus into its source, answers with ids that
# now belong to other issues and has nothing to fall back on here.
rm -rf /opt/corpus_hidden
python3 /tests/remap_corpus.py /opt/corpus /opt/corpus_hidden /opt/hidden/labels.json \
  /opt/hidden/labels_remapped.json /opt/hidden/id_map.json
chmod 700 /opt/corpus_hidden /opt/hidden && chmod 600 /opt/hidden/*
rm -rf /app/corpus
cp -a /opt/corpus_hidden /app/corpus
chown -R runner:runner /app/corpus
cp /opt/hidden/queries.json /app/queries/graded.json
run_once hidden /opt/run_hidden.json

# -P keeps the working directory off sys.path. The crate runs as `runner` and writes into
# /app, so without it an /app/pytest.py holding sys.exit(0) would be imported instead of the
# real module and the whole suite would pass for reward 1.
python -P -m pytest -o cache_dir=/tmp/pytest_cache \
  --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
exit 0
