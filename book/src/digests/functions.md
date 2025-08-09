# Digest: functions

## Core principles
- Rust requires you to specify the type of each parameter.
- The return type is declared after an `->`, and the last expression is returned implicitly.
- Functions can take ownership, borrow values immutably, or mutably.
- Top-level functions are declared with `fn`, and methods can be attached to structs via `impl`.

## Basic example
```rust
fn square(x: i32) -> i32 {
    x * x
}

fn main() {
    let result = square(4);
    println!("4 squared is {}", result);
}
```

The function `square` takes an `i32` and returns an `i32`.  
The return is implicit: no semicolon after `x * x`.

## With explicit return
```rust
fn square_explicit(x: i32) -> i32 {
    return x * x;
}
```

Use `return` only when you want to exit early or improve readability.

## With references (borrowing)
```rust
fn first_letter(s: &String) -> Option<char> {
    s.chars().next()
}
```

This borrows `s` (does not take ownership).  
The return type is `Option<char>` since `.next()` may return `None`.

## With mutable references
```rust
fn push_zero(v: &mut Vec<i32>) {
    v.push(0);
}
```

The `&mut` tells the compiler that this function modifies the vector.  
Rust enforces exclusive access to `v` while it is borrowed mutably.

## Generic functions
Rust does not support function overloading. Instead, use generics:
```rust
fn identity<T>(x: T) -> T {
    x
}
```

This works with any type `T`, as long as the usage allows it. You can also restrict what kinds of types `T` can be by adding trait bounds like `T: SomeTrait`.
We will introduce traits and trait bounds in another digest. As an illustration, consider this function:
```rust
fn add<T: Sum>(x: T, y: T) -> T {
    x + y
}
```
Here, only types `T` that implement the `Sum` trait can be used.

## Functions as values
Functions can be passed around like any other value:
```rust
fn greet(name: &str) {
    println!("Hello, {}!", name);
}

fn apply<F>(f: F)
where
    F: Fn(&str),
{
    f("world");
}

fn main() {
    apply(greet);
}
```

Functions implement the `Fn`, `FnMut`, or `FnOnce` traits.  
You can pass closures or named functions.

## Closures
Rust supports inline anonymous functions:
```rust
let add = |x: i32, y: i32| x + y;
println!("{}", add(2, 3));
```

Closures capture variables from the environment.

## Lifetimes in functions (advanced)
Sometimes you need to annotate lifetimes:
```rust
fn longer<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.len() > b.len() { a } else { b }
}
```

`'a` ensures the returned reference lives as long as both inputs.  
Useful when dealing with borrowed values in more complex code.

## Further reading
- [Rust Book – Functions](https://doc.rust-lang.org/book/ch03-03-how-functions-work.html)
- [Rust Reference – Functions](https://doc.rust-lang.org/reference/items/functions.html)
- [Rust by Example – Functions](https://doc.rust-lang.org/rust-by-example/fn.html)
