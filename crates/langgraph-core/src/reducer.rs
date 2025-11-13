//! State reduction and aggregation logic

use crate::{Result, State};
use std::fmt::Debug;

/// State aggregation and reduction trait
///
/// Reducers define how multiple state updates are combined.
pub trait Reducer<T>: Send + Sync + Debug
where
    T: Clone + Send + Sync,
{
    /// Reduce two values into one
    fn reduce(&self, left: T, right: T) -> Result<T>;

    /// Reduce multiple values
    fn reduce_many(&self, values: Vec<T>) -> Result<T> {
        if values.is_empty() {
            return Err(crate::Error::invalid_operation(
                "Cannot reduce empty collection",
            ));
        }

        let mut result = values[0].clone();
        for value in values.into_iter().skip(1) {
            result = self.reduce(result, value)?;
        }
        Ok(result)
    }
}

/// Replace reducer - always takes the right value
#[derive(Debug, Clone, Copy)]
pub struct ReplaceReducer;

impl<T> Reducer<T> for ReplaceReducer
where
    T: Clone + Send + Sync,
{
    fn reduce(&self, _left: T, right: T) -> Result<T> {
        Ok(right)
    }
}

/// Append reducer for vectors
#[derive(Debug, Clone, Copy)]
pub struct AppendReducer;

impl<T> Reducer<Vec<T>> for AppendReducer
where
    T: Clone + Send + Sync,
{
    fn reduce(&self, mut left: Vec<T>, mut right: Vec<T>) -> Result<Vec<T>> {
        left.append(&mut right);
        Ok(left)
    }
}

/// Merge reducer for hashmaps
#[derive(Debug, Clone, Copy)]
pub struct MergeReducer;

impl<K, V> Reducer<std::collections::HashMap<K, V>> for MergeReducer
where
    K: Clone + Send + Sync + Eq + std::hash::Hash,
    V: Clone + Send + Sync,
{
    fn reduce(
        &self,
        mut left: std::collections::HashMap<K, V>,
        right: std::collections::HashMap<K, V>,
    ) -> Result<std::collections::HashMap<K, V>> {
        left.extend(right);
        Ok(left)
    }
}

/// Sum reducer for numeric types
#[derive(Debug, Clone, Copy)]
pub struct SumReducer;

macro_rules! impl_sum_reducer {
    ($($t:ty),*) => {
        $(
            impl Reducer<$t> for SumReducer {
                fn reduce(&self, left: $t, right: $t) -> Result<$t> {
                    Ok(left + right)
                }
            }
        )*
    };
}

impl_sum_reducer!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);

/// Custom reducer with function
#[derive(Clone)]
pub struct FunctionReducer<T, F>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> Result<T> + Send + Sync + 'static,
{
    function: std::sync::Arc<F>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F> FunctionReducer<T, F>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> Result<T> + Send + Sync + 'static,
{
    /// Create a new function reducer
    pub fn new(function: F) -> Self {
        Self {
            function: std::sync::Arc::new(function),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, F> Debug for FunctionReducer<T, F>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> Result<T> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionReducer").finish()
    }
}

impl<T, F> Reducer<T> for FunctionReducer<T, F>
where
    T: Clone + Send + Sync,
    F: Fn(T, T) -> Result<T> + Send + Sync + 'static,
{
    fn reduce(&self, left: T, right: T) -> Result<T> {
        (self.function)(left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_reducer() {
        let reducer = ReplaceReducer;
        let result = reducer.reduce(1, 2).unwrap();
        assert_eq!(result, 2);
    }

    #[test]
    fn test_append_reducer() {
        let reducer = AppendReducer;
        let left = vec![1, 2];
        let right = vec![3, 4];
        let result = reducer.reduce(left, right).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_sum_reducer() {
        let reducer = SumReducer;
        let result = reducer.reduce(10, 20).unwrap();
        assert_eq!(result, 30);
    }

    #[test]
    fn test_function_reducer() {
        let reducer = FunctionReducer::new(|a: i32, b: i32| Ok(a * b));
        let result = reducer.reduce(3, 4).unwrap();
        assert_eq!(result, 12);
    }
}
