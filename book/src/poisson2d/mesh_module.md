# Mesh module

The `mesh` module defines the mesh data structure used throughout the Poisson 2D solver.  
It encapsulates the geometrical discretization of the domain, i.e., the set of vertices (points in space) and the list of finite elements that connect those vertices.  
Any details related to the definition of an elemen such as connectivity, shape functions, or element types are implemented in [`element.rs`](element_module.md) and simply referenced here.

## Mesh struct

The main struct in this module is `Mesh2d`. It holds:
- `vertices`: the coordinates of all mesh nodes as a `Vec<Point2<f64>>`. A vector (from the std lib) of `Point2<f64>` vectors (from `nalgebra`),
- `elements`: the list of finite elements as `Vec<Element>`,
- `element_type`: an `ElementType` enum indicating the type of all elements in the mesh (e.g., P1, Q1).

The `Element` struct and `ElementType` enum are defined in the [`element.rs`](element_module.md) module.

```rust
{{#include ../../../crates/poisson_2d/src/mesh.rs:mesh_struct}}
```

This struct is marked with `#[derive(Clone, Debug)]` to allow duplication and debug printing, which are useful for testing and inspecting the mesh.

## Mesh implementations

The implementation block provides:

- `new(...)`: a constructor that takes ownership of the vertex list, element list, and element type.
- `vertices(&self)`: returns an immutable slice of the mesh's vertices.
- `elements(&self)`: returns an immutable slice of the mesh's elements.
- `element_type(&self)`: returns a reference to the mesh's `ElementType`.


```rust
{{#include ../../../crates/poisson_2d/src/mesh.rs:mesh_impl}}
```

These accessor methods are intentionally read-only, ensuring the internal structure of the mesh cannot be mutated from outside without explicit intent.

## A simple unit test

The module includes a basic unit test to verify that:
- The mesh stores the correct number of vertices and elements,
- The `element_type` is stored and accessible correctly.

The test builds a simple unit square mesh with four vertices and one `Q1` element, then asserts the expected sizes and type.

```rust
{{#include ../../../crates/poisson_2d/src/mesh.rs:tests}}
```