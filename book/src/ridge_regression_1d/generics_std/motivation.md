# Generics: introduction

The aim of this section is to generalize our estimators so they work with any numeric type, not just `f64`. Rust makes this possible through generics and trait bounds. It's divided into 2 subsections:

1) [Generics & trait bounds](what_are_generics.md): Introduces generics and trait bounds. The floating-point type `f64` is replaced by a generic type `F` that can either be `f32` or `f64`. In Rust, generic types have no behavior by default, and we must tell the compiler which traits `F` should implement.
2) [Closed-form solution](closed_form_solution.md): Explains how to implement the closed-form solution with generics and traits.

In the [next final section](../structured_ndarray/motivation.md), we finally explore how to use the external crate `ndarray` for linear algebra, and how to incorporate additional Rust features such as optional values, pattern matching, and error handling.