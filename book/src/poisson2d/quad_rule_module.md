# Quadrature module

The quadrature module defines how we perform numerical integration over finite elements.  
In the FEM assembly, element integrals like
$$
\int_{\hat{\Omega}} (\nabla N_a)^T\,(\nabla N_b)\,|J|\, d\hat{\Omega}
\quad\text{and}\quad
\int_{\hat{\Omega}} N_a\, f \, |J| \, d\hat{\Omega}
$$
are approximated with quadrature rules (a set of points and weights) on the reference element. The mapping to physical space is handled elsewhere via the Jacobian.

## Quadrature struct

A quadrature rule is defined by:
- `points`: the evaluation points in reference coordinates, and
- `weights`: the corresponding weights.

```rust
{{#include ../../../crates/poisson_2d/src/quadrature.rs:quad_rule_struct}}
```

This lightweight container is used by the solver during element-level integration. For correctness, `points.len()` must equal `weights.len()`.

## Quadrature implementations

Two families of rules are provided:

- `triangle(order)`: simple rules on the reference triangle.  
  - `order = 1`: 1-point rule (centroid), total weight $\tfrac{1}{2}$ which matches the reference triangle area.  
  - `order = 2`: 3-point rule, exact for linear fields.

- `quadrilateral(n)`: tensor-product Gauss rules on the reference square $[-1,1]\times[-1,1]$.  
  - `n = 1`: 1-point Gauss rule at the center with weight $4$.  
  - `n = 2`: \(2\times2\) Gauss rule; points at $\pm \tfrac{1}{\sqrt{3}}$, each with weight $1$.

```rust
{{#include ../../../crates/poisson_2d/src/quadrature.rs:quad_rule_impl}}
```

**Notes**
- For triangles, the implementation panics for `order > 2` and similarly, quads panic for `n > 2`. Extend these if you need higher-order elements or exactness.
- The quadrilateral rule builds the 2D grid from the 1D Gauss abscissae $\pm 1/\sqrt{3}$ when `n = 2`.

## Simple tests

The tests check that the rule sizes match expectations for the provided configurations.

```rust
{{#include ../../../crates/poisson_2d/src/quadrature.rs:tests}}
```