# Adding tests

We can easily adapt the tests we implemented for the trait-based version of the optimizers. Here, we rely on pattern matching to check the constructors.

```rust
{{#include ../../../../crates/simple_optimizers_enums/src/optimizers.rs:tests}}
```