FROM public.ecr.aws/docker/library/rust:1.85-slim@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36

# pkg-config and libssl-dev are what the HTTP stack underneath hf-hub links against; without
# them the crate that reaches the Hub fails to build with an unhelpful OPENSSL_DIR error.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        pkg-config \
        libssl-dev \
        tmux \
        asciinema \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY corpus/ /app/corpus/
COPY queries/ /app/queries/
COPY search/ /app/search/

# Compile the dependency tree into the image. The task is solved with the network open, but
# the crates must not have to be fetched to build: only the agent's own sources recompile.
RUN cd /app/search && cargo build --release --locked && rm -rf /app/search/target/release/incremental

CMD ["/bin/bash"]
