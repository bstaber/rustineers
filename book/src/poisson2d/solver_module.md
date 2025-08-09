# Solver module

This module contains all the core logic required to assemble and solve the finite element system for the 2D Poisson equation.  
It supports both dense and sparse matrix formulations and includes routines for:
- Assembling the stiffness matrix and load vector from a given mesh and source function.
- Applying Dirichlet boundary conditions to either dense or sparse systems.
- Solving the resulting linear system using either a Cholesky decomposition (dense) or the Conjugate Gradient method (sparse).
- High-level “assemble-and-solve” functions for convenience.

We proceed in the same order as the implementation, explaining each part in detail.



## Mathematical background

We consider the Poisson problem in its weak form:

$$
\text{Find } u \in V \text{ such that:} \\
\int_{\Omega} \nabla u \cdot \nabla v \, dx = \int_{\Omega} f v \, dx, \quad \forall v \in V_0,
$$

where:
- $ \Omega $ is the computational domain.
- $ V $ is the trial space satisfying Dirichlet boundary conditions.
- $ V_0 $ is the test space with homogeneous Dirichlet conditions.
- $ f $ is the source term.

In the finite element method, the solution $ u_h $ is expanded in terms of basis functions $ \{ \phi_i \} $, and we obtain the linear system:

$$
A_{ij} = \int_{\Omega} \nabla \phi_i \cdot \nabla \phi_j \, dx, \quad
b_i = \int_{\Omega} f \, \phi_i \, dx.
$$

The stiffness matrix $A$ and load vector $b$ are computed by assembling contributions from each element in the mesh. Numerical integration is carried out using quadrature rules on a reference element, with transformations to the physical element through the Jacobian.



## Dense system assembly

The first function, `assemble_system_dense`, constructs the stiffness matrix $A$ and right-hand side vector $b$ for the given mesh and source term using a dense matrix representation.  
It:
1. Selects the appropriate reference element (`Tri3` or `Quad4`) based on the mesh element type.
2. Chooses a second-order quadrature rule to integrate element-level matrices.
3. Loops over each element, computes local stiffness `ke` and local load `fe`, and assembles them into the global system.

```rust
{{#include ../../../crates/poisson_2d/src/solver.rs:assemble_system_dense}}
```



## Sparse system assembly

The `assemble_system_sparse` function is analogous to the dense version, but uses a COO format during assembly and converts to CSR format for efficiency.  
This is better suited for large problems where most of the global matrix entries are zero.

```rust
{{#include ../../../crates/poisson_2d/src/solver.rs:assemble_system_sparse}}
```



## Applying Dirichlet boundary conditions (dense)

Dirichlet boundary conditions are enforced by:
1. Modifying the right-hand side vector to account for known values at boundary nodes.
2. Zeroing out the corresponding rows and columns in the stiffness matrix.
3. Setting diagonal entries to 1 and RHS entries to the Dirichlet values.

```rust
{{#include ../../../crates/poisson_2d/src/solver.rs:apply_dirichlet_dense}}
```



## Applying Dirichlet boundary conditions (sparse)

The sparse version performs similar operations, but with care to work directly with the CSR matrix structure.  
We iterate over rows and selectively zero out entries, preserving the sparse layout.

```rust
{{#include ../../../crates/poisson_2d/src/solver.rs:apply_dirichlet_sparse}}
```



## Dense solver

The dense solver uses a Cholesky factorization of the symmetric positive-definite stiffness matrix to compute the solution efficiently.

```rust
{{#include ../../../crates/poisson_2d/src/solver.rs:dense_solver}}
```



## Sparse solver

The sparse solver uses an iterative Conjugate Gradient (CG) method to solve the system, which is memory-efficient and scales better for large meshes.

```rust
{{#include ../../../crates/poisson_2d/src/solver.rs:sparse_solver}}
```



## High-level assemble-and-solve (dense)

This function combines the assembly, boundary condition application, and solve phases into a single call for dense systems.

```rust
{{#include ../../../crates/poisson_2d/src/solver.rs:assemble_and_solve_dense}}
```



## High-level assemble-and-solve (sparse)

Similarly, this high-level function handles all the steps for sparse systems in one call.

```rust
{{#include ../../../crates/poisson_2d/src/solver.rs:assemble_and_solve_sparse}}
```



## Unit tests

The tests check that both the dense and sparse assembly functions produce systems of the expected size for a simple 2x2 square mesh.

```rust
{{#include ../../../crates/poisson_2d/src/solver.rs:tests}}
```
