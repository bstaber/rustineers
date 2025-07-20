# Adding tests

In order to test our optimizers, we propose to have a look at how to implement tests and run them.

## How to write tests in Rust

Tests can be included in the same file as the code using the `#[cfg(test)]` module. Each test function is annotated with `#[test]`. Inside a test, you can use `assert_eq!`, `assert!`, or similar macros to validate expected behavior.

## What we test

We implemented a few tests to check:

- That the constructors return the expected variant with the correct parameters
- That the `step` method modifies weights as expected
- That repeated calls to `step` update the internal state correctly (e.g., momentum's velocity)

```rust
{{#include ../../../../crates/simple_optimizers_traits/src/optimizers.rs:tests}}
```

Some notes:
- This module is added in the same file where the optimizers are implemented.
- The line `use super::*;` tells the compiler to import all the stuff available in the module.

## How to run the tests

To run the tests from the command line, use:

```bash
cargo test
```

This will automatically find and execute all test functions in the project. You should see output like:

```
running 4 tests
test tests::test_gradient_descent_constructor ... ok
test tests::test_momentum_constructor ... ok
test tests::test_step_gradient_descent ... ok
test tests::test_step_momentum ... ok
```

If any test fails, Cargo will show which assertion failed and why.