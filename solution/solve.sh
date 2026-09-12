#!/bin/bash
set -euo pipefail

cp /solution/main.rs /app/search/src/main.rs
cd /app/search
cargo run --release
