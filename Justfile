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

# 🧱 Scaffold a new method
# Usage: just new-method lasso std
new-method name style:
    @echo "🔧 Scaffolding {{name}}_{{style}}..."
    cargo new --lib crates/{{name}}_{{style}}
    echo "mod helpers;\nmod one_liner;\n\npub use helpers::*;\npub use one_liner::*;" > crates/{{name}}_{{style}}/src/lib.rs
    mkdir -p book/src/{{name}}
    touch book/src/{{name}}/overview.md
    touch book/src/{{name}}/{{name}}_{{style}}.md
    echo "- [{{name}} Regression](./{{name}}/overview.md)\n  - [{{style}}](./{{name}}/{{name}}_{{style}}.md)" >> book/src/SUMMARY.md
    echo "✅ Done."
