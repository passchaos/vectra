use crate::core::MajorOrder;

pub fn compute_strides(shape: &[usize], major_order: MajorOrder) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

    match major_order {
        MajorOrder::RowMajor => {
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        MajorOrder::ColumnMajor => {
            for i in 1..shape.len() {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
        }
    }
    strides
}

pub fn flat_idx_to_indices(
    shape: &[usize],
    flat_idx: usize,
    major_order: MajorOrder,
) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
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

pub fn indices_to_flat_idx(strides: &[usize], indices: &[usize]) -> usize {
    strides.iter().zip(indices).map(|(&s, &i)| s * i).sum()
}

pub fn shape_indices_to_flat_idx(
    shape: &[usize],
    indices: &[usize],
    major_order: MajorOrder,
) -> usize {
    let strides = compute_strides(shape, major_order);
    indices_to_flat_idx(&strides, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_strides_idx() {
        let shape = vec![2, 3];

        let row_strides = compute_strides(&shape, MajorOrder::RowMajor);
        assert_eq!(row_strides, vec![3, 1]);
        let col_strides = compute_strides(&shape, MajorOrder::ColumnMajor);
        assert_eq!(col_strides, vec![1, 2]);

        let row_idx = indices_to_flat_idx(&row_strides, &[1, 1]);
        let row_idx1 = shape_indices_to_flat_idx(&shape, &[1, 1], MajorOrder::RowMajor);
        assert_eq!(row_idx, 4);
        assert_eq!(row_idx, row_idx1);

        let col_idx = indices_to_flat_idx(&col_strides, &[1, 1]);
        let col_idx1 = shape_indices_to_flat_idx(&shape, &[1, 1], MajorOrder::ColumnMajor);
        assert_eq!(col_idx, 3);
        assert_eq!(col_idx, col_idx1);

        let indices_1 = flat_idx_to_indices(&shape, row_idx, MajorOrder::RowMajor);
        let indices_2 = flat_idx_to_indices(&shape, col_idx, MajorOrder::ColumnMajor);

        assert_eq!(indices_1, indices_2);
        assert_eq!(indices_1, vec![1, 1]);
        assert_eq!(indices_2, vec![1, 1]);
    }
}
