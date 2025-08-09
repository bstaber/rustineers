# Element module

The `element` module defines the finite element types and the associated reference elements used by the solver.
It is responsible for everything specific to elements: types, connectivity, shape functions, gradients, and the jacobian needed to map between reference and physical coordinates.

This module is consumed by higher-level parts of the crate (mesh, quadrature, solver).

## Element struct and types

The crate currently supports two classical 2D elements:
- **P1**: 3-node linear triangle (Tri3),
- **Q1**: 4-node bilinear quadrilateral (Quad4).

`ElementType` encodes which type is used, and `Element` stores the connectivity via global node indices.

```rust
{{#include ../../../crates/poisson_2d/src/element.rs:elements}}
```

**Notes**
- `ElementType` derives `Clone`, `PartialEq`, `Eq`, and `Debug`, which makes pattern matching and testing straightforward.
- `Element` is a tiny container that holds the indices of the mesh vertices forming each element: geometry (actual coordinates) lives in the mesh.

## Reference element enum

The reference element encodes the canonical (parameter-space) version of each element type:
- `Tri3`: the unit reference triangle,
- `Quad4`: the unit reference square $[-1,1]\times[-1,1]$.

The method `num_nodes()` returns the number of nodes for each reference element.

```rust
{{#include ../../../crates/poisson_2d/src/element.rs:reference_elements}}
```

This abstraction separates element formulas (defined on reference coordinates) from the actual geometry (physical coordinates from the mesh).

## Reference element implementations

This block implements the core **finite element kinematics** on the reference element:

- `shape_functions(&Point2<f64>) -> Vec<f64>`  
  Returns the values of the shape functions $\{N_a\}$ at given local coordinates $(\xi, \eta)$.  
  These are used to interpolate fields (e.g., $u$) inside an element:  
  $$u(\xi,\eta) = \sum_a N_a(\xi,\eta)\, u_a.$$

- `shape_gradients(&Point2<f64>) -> Vec<Vector2<f64>>`  
  Returns the gradients on the reference element, i.e., $\partial N_a / \partial (\xi,\eta)$.  
  They are combined with the inverse Jacobian to obtain physical gradients $\nabla N_a$ during assembly.

- `jacobian(vertices_coordinates, local_coordinates) -> Matrix2<f64>`  
  Computes the Jacobian $J$ of the mapping from reference to physical coordinates.  
  For Tri3 this reduces to a constant matrix built from vertex differences; for Quad4 it is obtained by summing contributions of the shape function gradients weighted by vertex coordinates.


```rust
{{#include ../../../crates/poisson_2d/src/element.rs:reference_elements_impl}}
```

How it fits into assembly:
1. Evaluate shape functions $N_I$ (and gradients) at quadrature points in reference space.
2. Build the Jacobian $J$ and its determinant $|J|$.
3. Map reference gradients to physical gradients via $\nabla N_I = J^{-T} \nabla_{\xi\eta} N_I$.
4. Accumulate local stiffness and load contributions.

## Simple unit test

This smoke test checks that:
- the number of nodes reported by each reference element is correct,
- the shape function vectors have matching sizes.

```rust
{{#include ../../../crates/poisson_2d/src/element.rs:tests}}
```
