/// Multiplies a vector by a scalar.
///
/// # Arguments
///
/// * `scalar` - A scalar multiplier
/// * `vector` - A slice of f64 values
///
/// # Returns
///
/// A new vector containing the result of element-wise multiplication
///
/// # Why `&[f64]` instead of `Vec<f64]`?
///
/// We use a slice (`&[f64]`) because:
/// - It's more general: works with both arrays and vectors
/// - It avoids unnecessary ownership
/// - It's idiomatic and Clippy-compliant
// ANCHOR: mul_scalar_vec
pub fn mul_scalar_vec(scalar: f64, vector: &[f64]) -> Vec<f64> {
    vector.iter().map(|x| x * scalar).collect()
}
// ANCHOR_END: mul_scalar_vec

/// Subtracts two vectors element-wise.
///
/// # Arguments
///
/// * `a` - First input slice
/// * `b` - Second input slice
///
/// # Returns
///
/// A new `Vec<f64>` containing the element-wise difference `a[i] - b[i]`.
///
/// # Panics
///
/// Panics if `a` and `b` do not have the same length.
// ANCHOR: subtract_vectors
pub fn subtract_vectors(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
// ANCHOR_END: subtract_vectors

/// Dot product between two vectors.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
///
/// # Returns
///
/// The float value of the dot product.
///
/// # Panics
///
/// Panics if `a` and `b` do have the same length.
// ANCHOR: dot
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter().zip(b.iter()).map(|(xi, yi)| xi * yi).sum()
}
// ANCHOR_END: dot
