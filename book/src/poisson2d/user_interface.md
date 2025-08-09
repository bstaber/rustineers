# User interface

In this chapter, we start the other way around, and first have a look at the proposed user interfance. The full file `lib.rs` is provided below. It mostly contains an enum and a helper function for solving the 2D Poisson problem.

<details>
<summary>Click here to view: <b>lib.rs</b>.</summary>

```rust
{{#include ../../../crates/poisson_2d/src/lib.rs}}
```
</details>

## Solver type

In this example, we consider two types of finite element solvers:
- A dense solver which assembles dense matrices and uses a dense solver for the linear system.
- A sparse solver which assembles sparse matrices and uses a sparse solver for the linear system.

This is encoded by defining the following enumerate:

```rust
{{#include ../../../crates/poisson_2d/src/lib.rs:solver_type}}
```

It can be used to pick the solver type by passing `SolverType::Dense` or `SolverType::Sparse` to the helper function discussed below.

## Helper function

The helper function that the user should call is shown below. It takes the following input arguments:
- A mesh, given as a `Mesh2d` struct implemented in the [mesh.rs](mesh.md) module.
- The boundary nodes and the boundary function ($g$) for applying Dirichlet boundary conditions.
- The source function $f$.
- And the solver type (`SolverType::Dense` or `SolverType::Sparse`).

Based on the chosen solver type, the function either calls the dense or sparse methods thanks to pattern matching.

```rust
{{#include ../../../crates/poisson_2d/src/lib.rs:solve_poisson_2d}}
```