build for python:
RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release

build for rust:
cargo run --release

uv run pytest
uv run benchmark.py
uv run plot.py

## Benchmark 9.06 
Rust outperforms Python for large datasetets but still struggles for small datasets. Idea: Find out where rust spends its time with extensive performance counter.

## Benchmark 15.06
The reason was that simd didnt work. With new code it is better in every case.



can we assume contiguous arrays?????