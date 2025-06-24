# Cargo 101

> This chapter gives you everything you need to compile and run the examples in this book using Cargo, Rust’s official package manager and build tool.

## Creating a new project

To create a new Rust *library* project:

```bash
cargo new my_project --lib
```

To create a *binary* project (i.e., one with a `main.rs`):

```bash
cargo new my_project
```

## Building your project

Navigate into the project directory:

```bash
cd my_project
```

Then build it:

```bash
cargo build
```

This compiles your code in debug mode (faster builds, less optimization). You’ll find the output in `target/debug/`.

## Running your code

If it’s a binary crate (with a `main.rs`), you can run it:

```bash
cargo run
```

This compiles and runs your code in one go.

## Testing your code

To run tests in `lib.rs` or in any `#[cfg(test)]` module:

```bash
cargo test
```

## Cleaning build artifacts

Remove the `target/` directory and everything in it:

```bash
cargo clean
```

## Checking your code (without building)

```bash
cargo check
```

This quickly verifies your code compiles without generating the binary.

## Adding dependencies

To add dependencies, open `Cargo.toml` and add them under `[dependencies]`:

```toml
[dependencies]
ndarray = "0.15"
```

Or use the command line:

```bash
cargo add ndarray
```
