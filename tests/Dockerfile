# Verifier image. Separate container from the agent's: it sees only the paths listed in
# `artifacts`, and everything else it needs is baked here at build time.
FROM public.ecr.aws/docker/library/rust:1.85-slim@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36 AS rust

FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

# The agent's crate is rebuilt and run in here, so this image needs a Rust toolchain as well
# as pytest. Copying the toolchain out of the pinned Rust image avoids a build-time rustup.
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        gcc \
        g++ \
        libc6-dev \
        pkg-config \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.lock && rm /tmp/requirements.lock

# Compile the dependency tree, so the agent's crate rebuilds here with the network closed.
COPY skeleton/ /opt/skeleton/
RUN cd /opt/skeleton && cargo build --release --locked

# Populate the hf-hub cache with the pinned checkpoint. Grading runs with no network, so the
# model has to be on disk already; it cannot be shipped inside `environment/` because the
# weights file alone is 90 MB, past the 50 MB per-file limit, which is the reason this task
# needs the network in the first place.
COPY warmup.rs /tmp/warmup.rs
RUN cp /tmp/warmup.rs /opt/skeleton/src/main.rs \
    && cd /opt/skeleton && cargo run --release --locked \
    && rm /tmp/warmup.rs

# Put the stub back and drop the crate's own object files, so a stale artifact can never be
# mistaken for a build of the agent's sources. Cargo decides freshness by mtime, and uploaded
# sources can predate this image's layers.
COPY skeleton/src/main.rs /opt/skeleton/src/main.rs
RUN cd /opt/skeleton && cargo clean -p bulletin_search --release

# The rebuild runs as an unprivileged user, so give that user its own copy of the cache and a
# home it can write to.
RUN useradd --create-home --shell /usr/sbin/nologin runner \
    && mkdir -p /home/runner/.cache \
    && cp -a /root/.cache/huggingface /home/runner/.cache/huggingface \
    && chown -R runner:runner /home/runner

# Verifier-only assets. The corpus and the queries are served from here rather than collected
# from the agent, so both graded runs are provably over inputs the agent never touched.
COPY corpus/ /opt/corpus/
COPY queries/ /opt/queries/
COPY hidden/ /opt/hidden/
COPY golden/ /opt/golden/
RUN chmod 700 /opt/hidden /opt/golden

# The verifier's own code and fixtures. In separate mode Harbor does not upload tests/ into the
# image, so the image has to own /tests itself. test.sh closes the whole directory to the
# unprivileged build user before any agent-authored code runs, and abandons the run if it is
# still reachable, so the labels sitting under it are not exposed by being here.
COPY . /tests/

# Landing directories for the collected artifacts. Harbor copies /app/search/src/,
# Cargo.toml, Cargo.lock and /app/rankings.json in here, and the upload fails if the parent
# directory does not already exist.
RUN mkdir -p /app/search

WORKDIR /app
