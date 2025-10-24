//! Array Slicing Operations Examples
//!
//! This example demonstrates various slicing operations in Vectra,
//! including basic slicing, advanced indexing, and slice assignment.

use vectra::prelude::*;

fn main() {
    println!("=== Array Slicing Operations Examples ===");

    // Create a sample 3D array for slicing operations
    let data: Vec<i32> = (1..=24).collect();
    let arr = Array::from_vec(data, [2, 3, 4]);
    println!("Original 3D array (2x3x4):\n{}", arr);

    // 1. Basic range slicing
    println!("\n1. Basic range slicing:");

    // Create a 2D array for easier visualization
    let arr_2d = Array::from_vec((1..=12).collect::<Vec<i32>>(), [3, 4]);
    println!("2D array (3x4):\n{}", arr_2d);

    // Slice rows 0 to 1 (exclusive), all columns
    let slice_rows =
        arr_2d.slice::<SliceArgKind>([SliceArgKind::Range(0..2), SliceArgKind::RangeFull(..)]);
    println!("\nRows 0-1, all columns [0..2, ..]:\n{}", slice_rows);

    // Slice all rows, columns 1 to 3 (exclusive)
    let slice_cols =
        arr_2d.slice::<SliceArgKind>([SliceArgKind::RangeFull(..), SliceArgKind::Range(1..3)]);
    println!("\nAll rows, columns 1-2 [.., 1..3]:\n{}", slice_cols);

    // Slice specific region
    let slice_region = arr_2d.slice([1isize..3, 0isize..2]);
    println!("\nRows 1-2, columns 0-1 [1..3, 0..2]:\n{}", slice_region);

    // 2. Inclusive range slicing
    println!("\n2. Inclusive range slicing:");

    let slice_inclusive = arr_2d.slice([0isize..=1, 1isize..=2]);
    println!(
        "Rows 0-1 (inclusive), columns 1-2 (inclusive) [0..=1, 1..=2]:\n{}",
        slice_inclusive
    );

    // 3. Range from and range to
    println!("\n3. Range from and range to:");

    let slice_from =
        arr_2d.slice::<SliceArgKind>([SliceArgKind::RangeFrom(1..), SliceArgKind::RangeFull(..)]);
    println!("From row 1 to end [1.., ..]:\n{}", slice_from);

    let slice_to =
        arr_2d.slice::<SliceArgKind>([SliceArgKind::RangeTo(..2), SliceArgKind::RangeFull(..)]);
    println!("From start to row 2 (exclusive) [..2, ..]:\n{}", slice_to);

    let slice_to_inclusive = arr_2d.slice::<SliceArgKind>([
        SliceArgKind::RangeToInclusive(..=1),
        SliceArgKind::RangeFull(..),
    ]);
    println!(
        "From start to row 1 (inclusive) [..=1, ..]:\n{}",
        slice_to_inclusive
    );

    // 4. Negative indexing
    println!("\n4. Negative indexing:");

    let slice_negative = arr_2d.slice([-2isize.., -2isize..]);
    println!(
        "Last 2 rows, last 2 columns [-2.., -2..]:\n{}",
        slice_negative
    );

    let slice_neg_range = arr_2d.slice([0isize..-1, 1isize..-1]);
    println!(
        "All but last row, middle columns [0..-1, 1..-1]:\n{}",
        slice_neg_range
    );

    // 5. Array indexing (fancy indexing)
    println!("\n5. Array indexing (fancy indexing):");

    // Select specific rows and columns
    let fancy_slice = arr_2d.slice::<SliceArgKind>([
        vec![0, 2].into(), // Select rows 0 and 2
        vec![1, 3].into(), // Select columns 1 and 3
    ]);
    println!("Rows [0, 2], columns [1, 3]:\n{}", fancy_slice);

    // Non-contiguous selection
    let non_contiguous = arr_2d.slice::<SliceArgKind>([
        vec![2, 0, 1].into(), // Reorder rows
        vec![3, 0, 2].into(), // Reorder columns
    ]);
    println!(
        "\nReordered rows [2, 0, 1], columns [3, 0, 2]:\n{}",
        non_contiguous
    );

    // 6. Single element selection
    println!("\n6. Single element selection:");

    let single_row =
        arr_2d.slice::<SliceArgKind>([SliceArgKind::Array(vec![1]), SliceArgKind::RangeFull(..)]);
    println!("Single row 1 [[1], ..]:\n{}", single_row);

    let single_col =
        arr_2d.slice::<SliceArgKind>([SliceArgKind::RangeFull(..), SliceArgKind::Array(vec![2])]);
    println!("\nSingle column 2 [.., [2]]:\n{}", single_col);

    let single_element =
        arr_2d.slice::<SliceArgKind>([SliceArgKind::Array(vec![1]), SliceArgKind::Array(vec![2])]);
    println!("\nSingle element at [1, 2] [[1], [2]]:\n{}", single_element);

    // 7. 3D array slicing
    println!("\n7. 3D array slicing:");

    let slice_3d_first = arr.slice::<SliceArgKind>([
        SliceArgKind::Range(0..1),
        SliceArgKind::RangeFull(..),
        SliceArgKind::RangeFull(..),
    ]);
    println!(
        "First slice along axis 0 [0..1, .., ..]:\n{}",
        slice_3d_first
    );

    let slice_3d_middle = arr.slice::<SliceArgKind>([
        SliceArgKind::RangeFull(..),
        SliceArgKind::Range(1..2),
        SliceArgKind::RangeFull(..),
    ]);
    println!(
        "\nMiddle slice along axis 1 [.., 1..2, ..]:\n{}",
        slice_3d_middle
    );

    let slice_3d_partial = arr.slice::<SliceArgKind>([
        SliceArgKind::RangeFull(..),
        SliceArgKind::RangeFull(..),
        SliceArgKind::Range(1..3),
    ]);
    println!(
        "\nPartial slice along axis 2 [.., .., 1..3]:\n{}",
        slice_3d_partial
    );

    let slice_3d_complex = arr.slice([0isize..1, 1isize..3, 0isize..2]);
    println!(
        "\nComplex 3D slice [0..1, 1..3, 0..2]:\n{}",
        slice_3d_complex
    );

    // 8. Slice assignment
    println!("\n8. Slice assignment:");

    let mut mutable_arr = Array::from_vec((1..=12).collect::<Vec<i32>>(), [3, 4]);
    println!("Original array:\n{}", mutable_arr);

    // Create replacement values
    let replacement = Array::from_vec(vec![99, 88], [1, 2]);

    // Assign to a slice
    mutable_arr.slice_assign([1isize..2, 1isize..3], &replacement);
    println!(
        "\nAfter slice assignment [1..2, 1..3] = [99, 88]:\n{}",
        mutable_arr
    );

    // 9. Slice fill
    println!("\n9. Slice fill:");

    let mut fill_arr = Array::from_vec((1..=12).collect::<Vec<i32>>(), [3, 4]);
    println!("Original array:\n{}", fill_arr);

    // Fill a slice with a single value
    fill_arr.slice_fill([0isize..2, 2isize..4], 42);
    println!("\nAfter slice fill [0..2, 2..4] = 42:\n{}", fill_arr);

    // Fill with fancy indexing
    fill_arr.slice_fill::<SliceArgKind>([vec![0, 2].into(), vec![0, 1].into()], 77);
    println!(
        "\nAfter fancy slice fill [[0, 2], [0, 1]] = 77:\n{}",
        fill_arr
    );

    // 10. Padding operations
    println!("\n10. Padding operations:");

    let small_arr = Array::from_vec(vec![1, 2, 3, 4], [2, 2]);
    println!("Small array (2x2):\n{}", small_arr);

    // Pad with zeros: (top, bottom, left, right)
    let padded = small_arr.pad((1, 1, 1, 1), 0);
    println!("\nPadded with 1 zero on each side:\n{}", padded);

    // Asymmetric padding
    let asym_padded = small_arr.pad((0, 2, 1, 0), -1);
    println!("\nAsymmetric padding (0,2,1,0) with -1:\n{}", asym_padded);

    // 11. Advanced slicing patterns
    println!("\n11. Advanced slicing patterns:");

    let matrix = Array::from_vec((1..=16).collect::<Vec<i32>>(), [4, 4]);
    println!("4x4 matrix:\n{}", matrix);

    // Extract diagonal-like pattern
    let diagonal_slice =
        matrix.slice::<SliceArgKind>([vec![0, 1, 2, 3].into(), vec![0, 1, 2, 3].into()]);
    println!("\nDiagonal elements:\n{}", diagonal_slice);

    // Extract anti-diagonal
    let anti_diagonal =
        matrix.slice::<SliceArgKind>([vec![0, 1, 2, 3].into(), vec![3, 2, 1, 0].into()]);
    println!("\nAnti-diagonal elements:\n{}", anti_diagonal);

    // Extract corners
    let corners = matrix.slice::<SliceArgKind>([vec![0, 0, 3, 3].into(), vec![0, 3, 0, 3].into()]);
    println!("\nCorner elements:\n{}", corners);

    // 12. Strided slicing simulation
    println!("\n12. Strided slicing simulation:");

    let large_arr = Array::from_vec((0..20).collect::<Vec<i32>>(), [4, 5]);
    println!("Large array (4x5):\n{}", large_arr);

    // Simulate stride=2 by selecting every other element
    let strided_rows = large_arr.slice::<SliceArgKind>([
        SliceArgKind::Array(vec![0, 2]), // Every other row
        SliceArgKind::Range(0..5),       // All columns
    ]);
    println!("\nEvery other row:\n{}", strided_rows);

    let strided_cols = large_arr.slice::<SliceArgKind>([
        SliceArgKind::Range(0..4),          // All rows
        SliceArgKind::Array(vec![0, 2, 4]), // Every other column
    ]);
    println!("\nEvery other column:\n{}", strided_cols);

    // 13. Range combinations and edge cases
    println!("\n13. Range combinations and edge cases:");

    let test_arr = Array::from_vec((0..30).collect::<Vec<i32>>(), [5, 6]);
    println!("Test array (5x6):\n{}", test_arr);

    // Empty ranges
    let empty_range = test_arr.slice::<SliceArgKind>([
        SliceArgKind::Range(2..2), // Empty range
        SliceArgKind::RangeFull(..),
    ]);
    println!(
        "\nEmpty range [2..2, ..] (should be empty):\n{}",
        empty_range
    );

    // Full range equivalents
    let full_range1 = test_arr.slice::<SliceArgKind>([
        SliceArgKind::Range(0..5), // Equivalent to RangeFull
        SliceArgKind::RangeFull(..),
    ]);
    let full_range2 = test_arr.slice::<SliceArgKind>([
        SliceArgKind::RangeFull(..), // Direct RangeFull
        SliceArgKind::Range(0..6),
    ]);
    println!("\nFull range [0..5, ..] vs [.., 0..6] (should be identical):");
    println!("Range(0..5): shape {:?}", full_range1.shape());
    println!("RangeFull: shape {:?}", full_range2.shape());

    // Boundary ranges
    let boundary_start = test_arr.slice::<SliceArgKind>([
        SliceArgKind::Range(0..1), // First row only
        SliceArgKind::RangeFull(..),
    ]);
    println!("\nBoundary start [0..1, ..]:\n{}", boundary_start);

    let boundary_end = test_arr.slice::<SliceArgKind>([
        SliceArgKind::Range(4..5), // Last row only
        SliceArgKind::RangeFull(..),
    ]);
    println!("\nBoundary end [4..5, ..]:\n{}", boundary_end);

    // 14. Range type comparisons
    println!("\n14. Range type comparisons:");

    let demo_arr = Array::from_vec((1..=20).collect::<Vec<i32>>(), [4, 5]);
    println!("Demo array (4x5):\n{}", demo_arr);

    // Different ways to get first 2 rows
    let range_exclusive =
        demo_arr.slice::<SliceArgKind>([SliceArgKind::Range(0..2), SliceArgKind::RangeFull(..)]);
    let range_inclusive = demo_arr.slice::<SliceArgKind>([
        SliceArgKind::RangeInclusive(0..=1),
        SliceArgKind::RangeFull(..),
    ]);
    let range_to =
        demo_arr.slice::<SliceArgKind>([SliceArgKind::RangeTo(..2), SliceArgKind::RangeFull(..)]);
    let range_to_inclusive = demo_arr.slice::<SliceArgKind>([
        SliceArgKind::RangeToInclusive(..=1),
        SliceArgKind::RangeFull(..),
    ]);

    println!("\nFirst 2 rows using different range types:");
    println!("Range(0..2):\n{}", range_exclusive);
    println!("\nRangeInclusive(0..=1):\n{}", range_inclusive);
    println!("\nRangeTo(..2):\n{}", range_to);
    println!("\nRangeToInclusive(..=1):\n{}", range_to_inclusive);

    // Different ways to get last 2 rows
    let last_range_from =
        demo_arr.slice::<SliceArgKind>([SliceArgKind::RangeFrom(2..), SliceArgKind::RangeFull(..)]);
    let last_range_explicit =
        demo_arr.slice::<SliceArgKind>([SliceArgKind::Range(2..4), SliceArgKind::RangeFull(..)]);
    let last_range_inclusive = demo_arr.slice::<SliceArgKind>([
        SliceArgKind::RangeInclusive(2..=3),
        SliceArgKind::RangeFull(..),
    ]);

    println!("\nLast 2 rows using different range types:");
    println!("RangeFrom(2..):\n{}", last_range_from);
    println!("\nRange(2..4):\n{}", last_range_explicit);
    println!("\nRangeInclusive(2..=3):\n{}", last_range_inclusive);

    // 15. Practical Range applications
    println!("\n15. Practical Range applications:");

    let data_matrix = Array::from_vec((1..=100).collect::<Vec<i32>>(), [10, 10]);
    println!("Data matrix (10x10) - showing first 3x3:");
    let preview = data_matrix.slice([0isize..3, 0isize..3]);
    println!("{}", preview);

    // Extract header row (first row)
    let header =
        data_matrix.slice::<SliceArgKind>([SliceArgKind::Range(0..1), SliceArgKind::RangeFull(..)]);
    println!("\nHeader row [0..1, ..]:\n{}", header);

    // Extract data without header (all rows except first)
    let data_only = data_matrix
        .slice::<SliceArgKind>([SliceArgKind::RangeFrom(1..), SliceArgKind::RangeFull(..)]);
    println!("\nData without header [1.., ..] (showing first 3 rows):");
    let data_preview =
        data_only.slice::<SliceArgKind>([SliceArgKind::Range(0..3), SliceArgKind::RangeFull(..)]);
    println!("{}", data_preview);

    // Extract border elements
    let top_border =
        data_matrix.slice::<SliceArgKind>([SliceArgKind::Range(0..1), SliceArgKind::RangeFull(..)]);
    let bottom_border = data_matrix
        .slice::<SliceArgKind>([SliceArgKind::Range(9..10), SliceArgKind::RangeFull(..)]);
    let left_border =
        data_matrix.slice::<SliceArgKind>([SliceArgKind::RangeFull(..), SliceArgKind::Range(0..1)]);
    let right_border = data_matrix
        .slice::<SliceArgKind>([SliceArgKind::RangeFull(..), SliceArgKind::Range(9..10)]);

    println!("\nBorder extraction:");
    println!("Top border shape: {:?}", top_border.shape());
    println!("Bottom border shape: {:?}", bottom_border.shape());
    println!("Left border shape: {:?}", left_border.shape());
    println!("Right border shape: {:?}", right_border.shape());

    // Extract center region (excluding borders)
    let center = data_matrix.slice([1isize..9, 1isize..9]);
    println!("\nCenter region [1..9, 1..9] shape: {:?}", center.shape());
    let center_preview = center.slice([0isize..3, 0isize..3]);
    println!("Center preview (first 3x3):\n{}", center_preview);

    // 16. Range-based windowing
    println!("\n16. Range-based windowing:");

    let signal = Array::from_vec((0..20).collect::<Vec<i32>>(), [20]);
    println!("Signal data: {}", signal);

    // Sliding window simulation using ranges
    let window_size = 5;
    println!("\nSliding windows of size {}:", window_size);

    for i in 0..=(20 - window_size) {
        let window = signal.slice::<SliceArgKind>([SliceArgKind::Range(i..i + window_size)]);
        if i < 4 {
            // Show first few windows
            println!("Window {}: {}", i, window);
        } else if i == 4 {
            println!("... (showing first 4 windows)");
        }
    }

    // Overlapping ranges
    println!("\nOverlapping range examples:");
    let overlap1 = signal.slice::<SliceArgKind>([SliceArgKind::Range(0..10)]);
    let overlap2 = signal.slice::<SliceArgKind>([SliceArgKind::Range(5..15)]);
    let overlap3 = signal.slice::<SliceArgKind>([SliceArgKind::Range(10..20)]);

    println!("Range 0..10: {}", overlap1);
    println!("Range 5..15: {}", overlap2);
    println!("Range 10..20: {}", overlap3);

    println!("\n=== Array Slicing Operations Examples Complete ===");
}
