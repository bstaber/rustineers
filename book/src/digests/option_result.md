# Digest: option and result

This digest introduces two of the most important enums in Rust: `Option`
and `Result`. They are the foundation of safe error handling and absence
handling. Understanding how to use them is essential for writing robust
Rust programs.


## 1. The option type

`Option<T>` represents a value that may or may not be present.

``` rust
let maybe_num: Option<i32> = Some(42);
let nothing: Option<i32> = None;
```

### Extracting values

-   **unwrap**: take the value, panic if `None`.

``` rust
println!("{}", maybe_num.unwrap()); // 42
println!("{}", nothing.unwrap());   // panics
```

-   **expect**: like `unwrap` but with a custom panic message.

``` rust
println!("{}", maybe_num.expect("Should have a value"));
```

-   **if let**: execute code only when the option is `Some`.

``` rust
if let Some(x) = maybe_num {
    println!("Got {}", x);
}
```

-   **match**: handle both cases explicitly.

``` rust
match maybe_num {
    Some(x) => println!("Got {}", x),
    None => println!("Got nothing"),
}
```

-   **unwrap_or / unwrap_or_else**: provide a fallback value.

``` rust
let x = nothing.unwrap_or(0); // x = 0
```


## 2. The result type

`Result<T, E>` represents success (`Ok(T)`) or error (`Err(E)`).

``` rust
fn divide(a: i32, b: i32) -> Result<i32, &'static str> {
    if b == 0 {
        Err("cannot divide by zero")
    } else {
        Ok(a / b)
    }
}
```

### Extracting values

-   **unwrap / expect**: panic if `Err`.

``` rust
println!("{}", divide(4, 2).unwrap()); // 2
println!("{}", divide(4, 0).unwrap()); // panics
```

-   **if let**: handle only the success case.

``` rust
if let Ok(val) = divide(4, 2) {
    println!("Result is {}", val);
}
```

-   **match**: handle both success and error.

``` rust
match divide(4, 0) {
    Ok(val) => println!("Result is {}", val),
    Err(e) => println!("Error: {}", e),
}
```

-   **unwrap_or / unwrap_or_else**: provide a default on error.

``` rust
let val = divide(4, 0).unwrap_or(-1); // val = -1
```


## 3. The question mark operator

The `?` operator is the idiomatic way to propagate errors or `None`
values. It works only in functions returning `Option` or `Result`.

``` rust
fn read_number() -> Result<i32, std::num::ParseIntError> {
    let text = "42";
    let num: i32 = text.parse()?; // if parsing fails, return the error immediately
    Ok(num)
}
```

-   With `Option`, `None` will be returned early.
-   With `Result`, `Err(e)` will be returned early.


## 4. Rules of thumb

-   Use `unwrap` or `expect` only in tests, quick prototypes, or when
    failure is impossible.
-   Use `if let` when you only care about the success case.
-   Use `match` when you want to handle both success and failure
    explicitly.
-   Use `unwrap_or` when a fallback value makes sense.
-   Use `?` to simplify error propagation in functions that return
    `Option` or `Result`.


## 5. Putting it together

``` rust
fn safe_divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Option with match
    match safe_divide(4, 2) {
        Some(val) => println!("4/2 = {}", val),
        None => println!("division by zero"),
    }

    // Option with if let
    if let Some(val) = safe_divide(10, 5) {
        println!("10/5 = {}", val);
    }

    // Result with ?
    let number: i32 = "123".parse()?; // parse returns Result
    println!("Parsed: {}", number);

    Ok(())
}
```

This example illustrates how `Option` and `Result` can be used together
in practice.
