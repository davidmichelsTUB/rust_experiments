build for python:
uv run maturin develop --release

build for rust:
cargo run --release

uv run pytest
uv run benchmark.py
uv run plot.py
