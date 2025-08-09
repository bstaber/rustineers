# Poisson 2D solver in Rust

This crate implements a simple 2D finite element solver for the Poisson equation:

$$
- \nabla^2 u(x, y) = f(x, y) \quad \text{in} \; \Omega
$$
$$
u = g \quad \text{on} \; \partial \Omega
$$

where:
- $\Omega$ is a 2D domain discretized into finite elements (triangles or quadrangles),
- $f(x, y)$ is a given source function,
- $g$ is a Dirichlet boundary condition.

## Weak formulation

To solve the PDE using the finite element method, we first write its weak form:

Find $u \in \mathcal{H}_0$ such that:

$$
\int_{\Omega} \nabla u \cdot \nabla v \, dx = \int_{\Omega} f v \, dx
\quad \forall v \in V_0
$$

where $\mathcal{H}_0$ is the closure of the usual Sobolev space $\mathcal{H}$ in which we seek our solution. We then discretize the problem using finite element basis functions, leading to a linear system:

$$
A \mathbf{u} = \mathbf{b}
$$

with:
- **A**: global stiffness matrix (assembled from element contributions),
- **u**: vector of nodal values,
- **b**: load vector from the source term.

## Features
- Support for different element types (`P1`, `Q1`)
- Dense and sparse matrix assembly
- Dirichlet boundary condition handling

## Code Structure

At the end of the chapter, we obtain a small standalone crate with the following layout:
```text
├── Cargo.toml
└── src
    ├── element.rs
    ├── lib.rs
    ├── mesh.rs
    ├── quadrature.rs
    └── solver.rs
```

The crate is split into the following modules:

- [`element.rs`](src/element.rs): Defines finite element types and related data structures (e.g., connectivity, local stiffness).

- [`mesh.rs`](src/mesh.rs): Defines the `Mesh2d` structure, storing:
    - Vertex coordinates
    - Element connectivity
    - Element type

    Also provides accessors and utility methods for FEM assembly.

- [`quadrature.rs`](src/quadrature.rs): Implements quadrature (numerical integration) rules for computing element matrices.

- [`solver.rs`](src/solver.rs): Core numerical routines:
    - System assembly (dense & sparse versions)
    - Dirichlet boundary condition application
    - Linear system solver

- [`lib.rs`](src/lib.rs): Crate root where we re-export the main types and functions for easier use.

## Example

```rust
use poisson_2d::{solve_poisson_2d, Mesh2d, SolverType, DVector};
use poisson_2d::mesh::{Element, ElementType};
use nalgebra::Point2;

fn main() {
    // Build a tiny unit-square mesh with one Q1 quad (4 nodes, 1 element)
    let vertices = vec![
        Point2::new(0.0, 0.0), // 0
        Point2::new(1.0, 0.0), // 1
        Point2::new(1.0, 1.0), // 2
        Point2::new(0.0, 1.0), // 3
    ];
    // Only a single quad element
    let elements = vec![
        Element { indices: vec![0, 1, 2, 3] }
    ];
    let mesh = Mesh2d::new(vertices, elements, ElementType::Q1);

    // Define boundary nodes and functions
    let boundary_nodes = vec![0, 1];

    // Dirichlet boundary g(x,y) = 0
    let g = |_: f64, _: f64| 0.0;

    // Source term f(x,y) = x + y
    let f = |x: f64, y: f64| x + y;

    // Solve (choose Dense or Sparse)
    let u_dense: DVector<f64> =
        solve_poisson_2d(&mesh, &boundary_nodes, &g, &f, SolverType::Dense);

    let u_sparse: DVector<f64> =
        solve_poisson_2d(&mesh, &boundary_nodes, &g, &f, SolverType::Sparse);
}
```
