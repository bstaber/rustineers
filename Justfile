# Default: show available commands
default:
    @just --summary

# 🔧 Cargo build
build:
    cargo build --workspace

run:
    cargo run

# 🧪 Run tests
test:
    cargo test --workspace

# 🧽 Check formatting and linting
lint:
    cargo fmt
    cargo clippy --workspace -- -D warnings

# 🧼 Format all code
fmt:
    cargo fmt --all

# 📚 Build API docs
doc:
    cargo doc --workspace --open

# 📖 Build and serve mdBook
build-book:
    mdbook build book

serve-book:
    mdbook serve book