use crate::core::MajorOrder;

pub fn dyn_dim_to_static<const D: usize>(idx: &[usize]) -> [usize; D] {
    let mut result = [0; D];
    result.copy_from_slice(idx);

    result
}

pub fn compute_strides<const D: usize>(shape: [usize; D], major_order: MajorOrder) -> [usize; D] {
    let mut strides_d = vec![1; shape.len()];

    match major_order {
        MajorOrder::RowMajor => {
            for i in (0..shape.len() - 1).rev() {
                strides_d[i] = strides_d[i + 1] * shape[i + 1];
            }
        }
        MajorOrder::ColumnMajor => {
            for i in 1..shape.len() {
                strides_d[i] = strides_d[i - 1] * shape[i - 1];
            }
        }
    }

    let mut strides = [0; D];
    strides.copy_from_slice(&strides_d);

    strides
}

#[allow(unused)]
pub fn flat_idx_to_indices<const D: usize>(
    shape: [usize; D],
    flat_idx: usize,
    major_order: MajorOrder,
) -> [usize; D] {
    let mut indices = [0; D];
    let mut remaining = flat_idx;

    match major_order {
        MajorOrder::RowMajor => {
            for (i, &dim) in shape.iter().enumerate().rev() {
                indices[i] = remaining % dim;
                remaining /= dim;
            }
        }
        MajorOrder::ColumnMajor => {
            for (i, &dim) in shape.iter().enumerate() {
                indices[i] = remaining % dim;
                remaining /= dim;
            }
        }
    }

    indices
}

pub fn indices_to_flat_idx<const D: usize>(strides: [usize; D], indices: [usize; D]) -> usize {
    strides.into_iter().zip(indices).map(|(s, i)| s * i).sum()
}

pub fn shape_indices_to_flat_idx<const D: usize>(
    shape: [usize; D],
    indices: [usize; D],
    major_order: MajorOrder,
) -> usize {
    let strides = compute_strides(shape, major_order);
    indices_to_flat_idx(strides, indices)
}

pub fn negative_indices_to_positive<const D: usize>(indices: [isize; D]) -> [usize; D] {
    indices.map(|idx| negative_idx_to_positive::<D>(idx))
}

pub fn negative_idx_to_positive<const D: usize>(idx: isize) -> usize {
    if idx >= (D as isize) || idx <= -(D as isize + 1) {
        panic!("Axis out of bounds: rank= {}, idx= {}", D, idx);
    }

    if idx < 0 {
        (idx + D as isize) as usize
    } else {
        idx as usize
    }
}

/// Broadcast two shapes to a common shape
pub fn broadcast_shapes<const D: usize>(
    shape1: [usize; D],
    shape2: [usize; D],
) -> Result<[usize; D], String> {
    let mut result = [1; D];

    for i in 0..D {
        let idx = D - 1 - i;
        let dim1 = shape1[idx];
        let dim2 = shape2[idx];

        if dim1 == dim2 {
            result[idx] = dim1;
        } else if dim1 == 1 {
            result[idx] = dim2;
        } else if dim2 == 1 {
            result[idx] = dim1;
        } else {
            return Err(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                shape1, shape2
            ));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_strides_idx() {
        let shape = [2, 3];

        let row_strides = compute_strides(shape, MajorOrder::RowMajor);
        assert_eq!(row_strides, [3, 1]);
        let col_strides = compute_strides(shape, MajorOrder::ColumnMajor);
        assert_eq!(col_strides, [1, 2]);

        let row_idx = indices_to_flat_idx(row_strides, [1, 1]);
        let row_idx1 = shape_indices_to_flat_idx(shape, [1, 1], MajorOrder::RowMajor);
        assert_eq!(row_idx, 4);
        assert_eq!(row_idx, row_idx1);

        let col_idx = indices_to_flat_idx(col_strides, [1, 1]);
        let col_idx1 = shape_indices_to_flat_idx(shape, [1, 1], MajorOrder::ColumnMajor);
        assert_eq!(col_idx, 3);
        assert_eq!(col_idx, col_idx1);

        let indices_1 = flat_idx_to_indices(shape, row_idx, MajorOrder::RowMajor);
        let indices_2 = flat_idx_to_indices(shape, col_idx, MajorOrder::ColumnMajor);

        assert_eq!(indices_1, indices_2);
        assert_eq!(indices_1, [1, 1]);
        assert_eq!(indices_2, [1, 1]);
    }

    #[test]
    fn test_shape_broadcast() {
        let shape1 = [2, 3];
        let shape2 = [1, 3];
        let result = broadcast_shapes(shape1, shape2).unwrap();
        assert_eq!(result, [2, 3]);

        let shape1 = [2, 3];
        let shape2 = [1, 3];
        let result = broadcast_shapes(shape1, shape2).unwrap();
        assert_eq!(result, [2, 3]);

        let shape1 = [2, 3];
        let shape2 = [2, 3];
        let result = broadcast_shapes(shape1, shape2).unwrap();
        assert_eq!(result, [2, 3]);

        let shape1 = [2, 3];
        let shape2 = [2, 4];
        let result = broadcast_shapes(shape1, shape2);
        assert!(result.is_err());
    }
}
