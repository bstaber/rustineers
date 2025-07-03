# Contributing to rustineers

## Project overview

The project is structured as an mdBook and a Rust workspace. It includes:

- `book/`: Markdown chapters
- `crates/`: Rust source code examples
- `justfile`: Task shortcuts using the `just` command runner

## Getting started

Besides Rust, you will need:

- Rust (https://www.rust-lang.org/tools/install)
- mdBook (https://rust-lang.github.io/mdBook/)
- mdBook-admonish (https://github.com/tommilligan/mdbook-admonish)
- `just` (https://github.com/casey/just)

Use `just` to simplify common tasks. To preview the book locally:

```bash
just serve-book
```

This will serve the book at http://localhost:3000.

Other useful command:

- `just lint` – Check formatting and linting
