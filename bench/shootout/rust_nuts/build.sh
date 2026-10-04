#!/usr/bin/env bash
# Builds the nuts-rs program into build/nutsrs/target/release/rust_nuts.
# CARGO_HOME is kept inside the worktree (the user's ~/.cargo is read-only
# in the sandbox this was run in). Regenerates src/dynpois.rs first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
export CARGO_HOME="${CARGO_HOME_SHOOTOUT:-$ROOT/build/cargo_home}"
export CARGO_TARGET_DIR="$ROOT/build/nutsrs/target"
python3 "$HERE/make_kernel.py" >/dev/null
cd "$HERE"
cargo build --release "$@" || cargo build --release "$@" || cargo build --release "$@"
