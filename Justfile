default:
    @just --summary

build:
    cargo build --workspace

clean:
    cargo clean

test:
    cargo test --workspace

lint:
    cargo fmt --all
    cargo clippy --workspace -- -D warnings

doc:
    cargo doc --workspace --open

build-book:
    mdbook build book

serve-book:
    mdbook serve book