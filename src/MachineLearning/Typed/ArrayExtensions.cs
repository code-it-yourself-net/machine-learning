﻿// Machine Learning Utils
// File name: ArrayExtensions.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace MachineLearning.Typed;

public static class ArrayExtensions
{

#if DEBUG
    private const string NumberOfColumnsMustBeEqualToNumberOfColumnsMsg = "The number of columns of the first matrix must be equal to the number of columns of the second matrix.";
    private const string NumberOfRowsMustBeEqualToNumberOfRowsMsg = "The number of rows of the first matrix must be equal to the number of rows of the second matrix.";
    private const string NumberOfColumnsMustBeEqualToNumberOfRowsMsg = "The number of columns of the first matrix must be equal to the number of rows of the second matrix.";
    private const string NumberOfRowsMustBeEqualToOneMsg = "The number of rows of the second matrix must be equal to 1.";
    private const string InvalidSizesMsg = "The sizes of the matrices are not compatible for elementwise multiplication.";
#endif

    #region Zeros, Ones, and Random

    /// <summary>
    /// Creates a new matrix filled with ones, with the same dimensions as the specified matrix.
    /// </summary>
    /// <param name="source">The matrix used to determine the dimensions of the new matrix.</param>
    /// <returns>A new matrix filled with ones.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] AsOnes(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = 1;
            }
        }

        return res;
    }

    /// <summary>
    /// Creates a new matrix filled with ones, with the same dimensions as the specified matrix.
    /// </summary>
    /// <param name="source">The matrix used to determine the dimensions of the new matrix.</param>
    /// <returns>A new matrix filled with ones.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] AsOnes(this float[,,,] source)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        res[i, j, k, l] = 1;
                    }
                }
            }
        }

        return res;
    }

    /// <summary>
    /// Creates a new matrix filled with ones, with the same dimensions as the specified matrix.
    /// </summary>
    /// <param name="source">The matrix used to determine the dimensions of the new matrix.</param>
    /// <returns>A new matrix filled with ones.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] AsOnes(this float[] source)
    {
        int length = source.GetLength(0);
        float[] res = new float[length];

        for (int i = 0; i < length; i++)
        {
            res[i] = 1;
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] AsZeroOnes(this float[,] source, float onesProbability, Random random)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] res = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                if (random.NextDouble() < onesProbability)
                {
                    res[i, j] = 1;
                }
            }
        }
        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] AsZeroOnes(this float[,,,] source, float onesProbability, Random random)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];
        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        if (random.NextDouble() < onesProbability)
                        {
                            res[i, j, k, l] = 1;
                        }
                    }
                }
            }
        }
        return res;
    }

    #endregion

    #region Slices, Rows, and Columns

    /// <summary>
    /// Gets a submatrix containing the specified column from the current matrix. The shape is [rows, 1].
    /// </summary>
    /// <param name="column"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] GetColumn(this float[,] source, int column)
    {
        int rows = source.GetLength(0);

        // Create an array to store the column.
        float[,] res = new float[rows, 1];

        for (int i = 0; i < rows; i++)
        {
            // Access each element in the specified column.
            res[i, 0] = source[i, column];
        }

        return res;
    }

    /// <summary>
    /// Gets a submatrix containing the specified range of columns from the current matrix. The shape is [rows, range].
    /// </summary>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] GetColumns(this float[,] source, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(source.GetLength(1));

        int rows = source.GetLength(0);
        float[,] res = new float[rows, length];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < length; j++)
            {
                res[i, j] = source[i, j + offset];
            }
        }

        return res;
    }

    /// <summary>
    /// Gets a row from the matrix.
    /// </summary>
    /// <param name="row">The index of the row to retrieve.</param>
    /// <returns>A new <see cref="Matrix"/> object representing the specified row.</returns>
    /// <remarks>
    /// The returned row is a new instance of the <see cref="Matrix"/> class and has the same number of columns as the original matrix.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] GetRow(this float[,] source, int row)
    {
        int columns = source.GetLength(1);

        // Create an array to store the row.
        float[] res = new float[columns];
        for (int i = 0; i < columns; i++)
        {
            // Access each element in the specified row.
            res[i] = source[row, i];
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,] GetRow(this float[,,,] source, int row)
    {
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        // Create an array to store the row.
        float[,,] res = new float[dim2, dim3, dim4];
        for (int i = 0; i < dim2; i++)
        {
            for (int j = 0; j < dim3; j++)
            {
                for (int k = 0; k < dim4; k++)
                {
                    // Access each element in the specified row.
                    res[i, j, k] = source[row, i, j, k];
                }
            }
        }

        return res;

    }

    /// <summary>
    /// Gets a submatrix containing the specified range of rows from the current matrix.
    /// </summary>
    /// <param name="range">The range of rows to retrieve.</param>
    /// <returns>A new <see cref="Matrix"/> object representing the submatrix.</returns>
    /// <remarks>
    /// The returned rows are a new instance of the <see cref="Matrix"/> class and have the same number of columns as the original matrix.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] GetRows(this float[,] source, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(source.GetLength(0));

        int columns = source.GetLength(1);
        float[,] res = new float[length, columns];

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = source[i + offset, j];
            }
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] GetRows(this float[,,,] source, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(source.GetLength(0));

        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[length, dim2, dim3, dim4];

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        res[i, j, k, l] = source[i + offset, j, k, l];
                    }
                }
            }
        }

        return res;

    }

    /// <summary>
    /// Sets the values of a specific row in the matrix.
    /// </summary>
    /// <param name="rowIndex">The index of the row to set.</param>
    /// <param name="row">The matrix containing the values to set.</param>
    /// <exception cref="Exception">Thrown when the number of columns in the specified matrix is not equal to the number of columns in the current matrix.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SetRow(this float[,] source, int rowIndex, float[] row)
    {

#if DEBUG
        if (row.GetLength(0) != source.GetLength(1))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);
#endif

        int columns = source.GetLength(1);
        for (int col = 0; col < columns; col++)
        {
            source[rowIndex, col] = row[col];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SetRow(this float[,,,] source, int rowIndex, float[,,] row)
    {
        Debug.Assert(rowIndex >= 0 && rowIndex < source.GetLength(0), "Row index out of bounds.");

        //int rows = source.GetLength(0);
        int cols = source.GetLength(1);
        int depths = source.GetLength(2);
        int channels = source.GetLength(3);

        for (int col = 0; col < cols; col++)
        {
            for (int depth = 0; depth < depths; depth++)
            {
                for (int channel = 0; channel < channels; channel++)
                {
                    source[rowIndex, col, depth, channel] = row[col, depth, channel];
                }
            }
        }
    }

    #endregion

    #region Aggregations

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(this float[,] source)
    {
        float max = float.MinValue;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                max = Math.Max(max, source[row, col]);
            }
        }
        return max;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(this float[,,,] source)
    {
        float max = float.MinValue;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        max = Math.Max(max, source[i, j, k, l]);
                    }
                }
            }
        }

        return max;
    }

    /// <summary>
    /// Calculates the mean of all elements in the matrix.
    /// </summary>
    /// <returns>The mean of all elements in the matrix.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Mean(this float[,] source) => source.Sum() / source.Length;

    /// <summary>
    /// Calculates the mean of all elements in the matrix.
    /// </summary>
    /// <returns>The mean of all elements in the matrix.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Mean(this float[,,,] source) => source.Sum() / source.Length;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(this float[,] source)
    {
        float min = float.MaxValue;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                min = Math.Min(min, source[row, col]);
            }
        }
        return min;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(this float[,,,] source)
    {
        float min = float.MaxValue;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for(int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        min = Math.Min(min, source[i, j, k, l]);
                    }
                }
            }
        }


        return min;
    }

    /// <summary>
    /// Calculates the standard deviation.
    /// </summary>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Std(this float[,] source)
    {
        float mean = source.Mean();
        float sum = 0;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                sum += MathF.Pow(source[i, j] - mean, 2);
            }
        }

        return (float)Math.Sqrt(sum / source.Length);
    }

    /// <summary>
    /// Calculates the standard deviation.
    /// </summary>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Std(this float[,,,] source)
    {
        float mean = source.Mean();
        float sum = 0;

        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        sum += MathF.Pow(source[i, j, k, l] - mean, 2);
                    }
                }
            }
        }

        return (float)Math.Sqrt(sum / source.Length);
    }

    /// <summary>
    /// Calculates the sum of all elements in the matrix.
    /// </summary>
    /// <returns>The sum of all elements in the matrix.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(this float[,] source)
    {
        // Sum over all elements.
        float sum = 0;
        int rows = source.GetLength(0);
        int cols = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                sum += source[row, col];
            }
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(this float[,,,] source)
    {
        // Sum over all elements.
        float sum = 0;

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);
        int depth = source.GetLength(2);
        int channels = source.GetLength(3);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                for (int k = 0; k < depth; k++)
                {
                    for (int l = 0; l < channels; l++)
                    {
                        sum += source[i, j, k, l];
                    }
                }
            }
        }

        return sum;
    }

    /// <summary>
    /// Summary of rows in one row.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] SumByRows(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[] res = new float[columns];

        for (int col = 0; col < columns; col++)
        {
            float sum = 0;
            for (int row = 0; row < rows; row++)
            {
                sum += source[row, col];
            }
            res[col] = sum;
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] AvgByRows(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[] res = new float[columns];

        for (int col = 0; col < columns; col++)
        {
            float sum = 0;
            for (int row = 0; row < rows; row++)
            {
                sum += source[row, col];
            }
            res[col] = sum / rows;
        }

        return res;
    }

    #endregion

    #region Operations with scalar

    /// <summary>
    /// Adds a scalar value to each element of the matrix.
    /// </summary>
    /// <param name="scalar">The scalar value to add.</param>
    /// <returns>A new matrix with the scalar added to each element.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Add(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] + scalar;
            }
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddInPlace(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                source[row, col] += scalar;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddInPlace(this float[,,,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        int depth = source.GetLength(2);
        int channels = source.GetLength(3);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                for (int k = 0; k < depth; k++)
                {
                    for (int l = 0; l < channels; l++)
                    {
                        source[i, j, k, l] += scalar;
                    }
                }
            }
        }

    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DivideInPlace(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                source[row, col] /= scalar;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DivideInPlace(this float[,,,] source, float scalar)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        for (int d1 = 0; d1 < dim1; d1++)
        {
            for (int d2 = 0; d2 < dim2; d2++)
            {
                for (int d3 = 0; d3 < dim3; d3++)
                {
                    for (int d4 = 0; d4 < dim4; d4++)
                    {
                        source[d1, d2, d3, d4] /= scalar;
                    }
                }
            }
        }

    }

    /// <summary>
    /// Multiplies each element of the matrix by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply.</param>
    /// <returns>A new matrix with each element multiplied by the scalar value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Multiply(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] * scalar;
            }
        }

        return res;
    }

    /// <summary>
    /// Multiplies each element of the matrix by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply.</param>
    /// <returns>A new matrix with each element multiplied by the scalar value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Multiply(this float[,,,] source, float scalar)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        for (int d1 = 0; d1 < dim1; d1++)
        {
            for (int d2 = 0; d2 < dim2; d2++)
            {
                for (int d3 = 0; d3 < dim3; d3++)
                {
                    for (int d4 = 0; d4 < dim4; d4++)
                    {
                        res[d1, d2, d3, d4] = source[d1, d2, d3, d4] * scalar;
                    }
                }
            }
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MultiplyInPlace(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                source[row, column] *= scalar;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Divide(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] / scalar;
            }
        }

        return res;
    }

    /// <summary>
    /// Raises each element of the matrix to the specified power.
    /// </summary>
    /// <param name="scalar">The power to raise each element to.</param>
    /// <returns>A new matrix with each element raised to the specified power.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Power(this float[,] source, int scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = MathF.Pow(source[row, col], scalar);
            }
        }

        return res;
    }

    #endregion

    #region Operations with matrix

    /// <summary>
    /// Adds a row to the current matrix by elementwise addition with the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to add as a row.</param>
    /// <returns>A new matrix with the row added.</returns>
    /// <exception cref="Exception">Thrown when the number of columns in the specified matrix is not equal to the number of columns in the current matrix, or when the number of rows of the specified matrix is not equal to 1.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] AddRow(this float[,] source, float[] matrix)
    {

#if DEBUG
        if (source.GetLength(1) != matrix.GetLength(0))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);
#endif

        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        //float[,] rowArray = row.Array;

        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = source[row, col] + matrix[col];
            }
        }

        return res;
    }

    /// <summary>
    /// Clips the values of the matrix in-place between the specified minimum and maximum values.
    /// </summary>
    /// <param name="min">The minimum value to clip the matrix elements to.</param>
    /// <param name="max">The maximum value to clip the matrix elements to.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ClipInPlace(this float[,] source, float min, float max)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                source[row, col] = MathF.Max(min, MathF.Min(max, source[row, col]));
            }
        }
    }

    /// <summary>
    /// Multiplies the current matrix with another matrix using the dot product.
    /// </summary>
    /// <param name="matrix">The matrix to multiply with.</param>
    /// <returns>A new matrix that is the result of the dot product multiplication.</returns>
    /// <exception cref="Exception">Thrown when the number of columns in the current matrix is not equal to the number of rows in the specified matrix.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] MultiplyDot(this float[,] source, float[,] matrix)
    {

#if DEBUG
        if (source.GetLength(1) != matrix.GetLength(0))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfRowsMsg);
#endif

        int matrixColumns = matrix.GetLength(1);

        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] res = new float[rows, matrixColumns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < matrixColumns; j++)
            {
                float sum = 0;
                for (int k = 0; k < columns; k++)
                {
                    sum += source[i, k] * matrix[k, j];
                }
                res[i, j] = sum;
            }
        }

        return res;
    }

    /// <summary>
    /// Performs elementwise multiplication between this matrix and another matrix.
    /// </summary>
    /// <param name="matrix">The matrix to multiply elementwise with.</param>
    /// <returns>A new matrix resulting from the elementwise multiplication.</returns>
    /// <remarks>
    /// Multiplies each element of the matrix with the corresponding element of another matrix.
    /// If the dimensions of the two matrices are not the same, the smaller matrix is broadcasted to match the larger matrix.
    /// If the size of this matrix is (a * b), and the size of matrix is (c * d), then the resulting size is (max(a,c) * max(b,d))
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] MultiplyElementwise(this float[,] source, float[,] matrix)
    {
        int thisRows = source.GetLength(0);
        int thisColumns = source.GetLength(1);
        int matrixRows = matrix.GetLength(0);
        int matrixColumns = matrix.GetLength(1);

        int maxRows = Math.Max(thisRows, matrixRows);
        int maxColumns = Math.Max(thisColumns, matrixColumns);

#if DEBUG
        // Make sure that the analogous sizes of both matrices are multiples of each other or - especially - are equal
        if (maxRows % thisRows != 0 || maxRows % matrixRows != 0 || maxColumns % thisColumns != 0 || maxColumns % matrixColumns != 0)
        {
            throw new Exception(InvalidSizesMsg);
        }
#endif

        float[,] res = new float[maxRows, maxColumns];

        for (int row = 0; row < maxRows; row++)
        {
            for (int col = 0; col < maxColumns; col++)
            {
                float thisValue = source[row % thisRows, col % thisColumns];
                float matrixValue = matrix[row % matrixRows, col % matrixColumns];
                res[row, col] = thisValue * matrixValue;
            }
        }

        return res;
    }

    /// <summary>
    /// Performs elementwise multiplication between this matrix and another matrix.
    /// </summary>
    /// <param name="matrix">The matrix to multiply elementwise with.</param>
    /// <returns>A new matrix resulting from the elementwise multiplication.</returns>
    /// <remarks>
    /// Multiplies each element of the matrix with the corresponding element of another matrix.
    /// If the dimensions of the two matrices are not the same, the smaller matrix is broadcasted to match the larger matrix.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] MultiplyElementwise(this float[,,,] source, float[,,,] matrix)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        int maxDim1 = Math.Max(dim1, matrix.GetLength(0));
        int maxDim2 = Math.Max(dim2, matrix.GetLength(1));
        int maxDim3 = Math.Max(dim3, matrix.GetLength(2));
        int maxDim4 = Math.Max(dim4, matrix.GetLength(3));

        float[,,,] res = new float[maxDim1, maxDim2, maxDim3, maxDim4];

        for (int d1 = 0; d1 < maxDim1; d1++)
        {
            for (int d2 = 0; d2 < maxDim2; d2++)
            {
                for (int d3 = 0; d3 < maxDim3; d3++)
                {
                    for (int d4 = 0; d4 < maxDim4; d4++)
                    {
                        float thisValue = source[d1 % dim1, d2 % dim2, d3 % dim3, d4 % dim4];
                        float matrixValue = matrix[d1 % matrix.GetLength(0), d2 % matrix.GetLength(1), d3 % matrix.GetLength(2), d4 % matrix.GetLength(3)];
                        res[d1, d2, d3, d4] = thisValue * matrixValue;
                    }
                }
            }
        }

        return res;

    }

    /// <summary>
    /// Performs elementwise multiplication between this matrix and another matrix.
    /// </summary>
    /// <param name="matrix">The matrix to multiply elementwise with.</param>
    /// <returns>A new matrix resulting from the elementwise multiplication.</returns>
    /// <remarks>
    /// Multiplies each element of the matrix with the corresponding element of another matrix.
    /// If the dimensions of the two matrices are not the same, the smaller matrix is broadcasted to match the larger matrix.
    /// If the size of this matrix is (a), and the size of matrix is (c * d), then the resulting size is (max(a,c) * d)
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] MultiplyElementwise(this float[] source, float[,] matrix)
    {
        int thisColumns = source.GetLength(0);
        int matrixRows = matrix.GetLength(0);
        int matrixColumns = matrix.GetLength(1);

        int maxColumns = Math.Max(thisColumns, matrixColumns);

#if DEBUG
        // Make sure that the analogous sizes of both matrices are multiples of each other or - especially - are equal
        if (maxColumns % thisColumns != 0 || maxColumns % matrixColumns != 0)
        {
            throw new Exception(InvalidSizesMsg);
        }
#endif

        float[,] res = new float[matrixRows, maxColumns];

        for (int row = 0; row < matrixRows; row++)
        {
            for (int col = 0; col < maxColumns; col++)
            {
                float thisValue = source[col % thisColumns];
                float matrixValue = matrix[row % matrixRows, col % matrixColumns];
                res[row, col] = thisValue * matrixValue;
            }
        }

        return res;
    }

    /// <summary>
    /// Subtracts the elements of the specified matrix from the current matrix.
    /// </summary>
    /// <param name="matrix">The matrix to subtract.</param>
    /// <returns>A new matrix with the elements subtracted.</returns>
    /// <exception cref="Exception">Thrown when the number of rows in the specified matrix is not equal to the number of rows in the current matrix, or when the number of columns in the specified matrix is not equal to the number of columns in the current matrix.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Subtract(this float[,] source, float[,] matrix)
    {
#if DEBUG
        if (source.GetLength((int)Dimension.Rows) != matrix.GetLength((int)Dimension.Rows))
            throw new Exception(NumberOfRowsMustBeEqualToNumberOfRowsMsg);

        if (source.GetLength((int)Dimension.Columns) != matrix.GetLength((int)Dimension.Columns))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);
#endif

        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = source[i, j] - matrix[i, j];
            }
        }

        return res;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Subtract(this float[,,,] source, float[,,,] matrix)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        for (int d1 = 0; d1 < dim1; d1++)
        {
            for (int d2 = 0; d2 < dim2; d2++)
            {
                for (int d3 = 0; d3 < dim3; d3++)
                {
                    for (int d4 = 0; d4 < dim4; d4++)
                    {
                        res[d1, d2, d3, d4] = source[d1, d2, d3, d4] - matrix[d1, d2, d3, d4];
                    }
                }
            }
        }

        return res;
    }

    #endregion

    #region Matrix operations and functions

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int[] Argmax(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        int[] array = new int[rows];

        for (int row = 0; row < rows; row++)
        {
            float max = float.MinValue;
            int maxIndex = 0;
            for (int col = 0; col < columns; col++)
            {
                float value = source[row, col];
                if (value > max)
                {
                    max = value;
                    maxIndex = col;
                }
            }
            array[row] = maxIndex;
        }

        return array;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Log(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = MathF.Log(source[i, j]);
            }
        }

        return res;
    }

    /// <summary>
    /// Applies the sigmoid function to each element of the matrix.
    /// </summary>
    /// <returns>A new matrix with each element transformed by the sigmoid function with the same dimensions as the original matrix.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Sigmoid(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = 1 / (1 + MathF.Exp(-source[i, j]));
            }
        }

        return res;
    }

    /// <summary>
    /// Applies the softmax function to the matrix.
    /// </summary>
    /// <returns>A new matrix with softmax-applied values.</returns>
    /// <remarks>Softmax formula: <c>exp(x) / sum(exp(x))</c>.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Softmax(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        float[,] expCache = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                expCache[i, j] = MathF.Exp(source[i, j]);
            }

            float sum = 0;
            for (int j = 0; j < columns; j++)
            {
                sum += expCache[i, j];
            }

            for (int j = 0; j < columns; j++)
            {
                res[i, j] = expCache[i, j] / sum;
            }
        }

        //for (int i = 0; i < rows; i++)
        //{
        //    float sum = 0;
        //    for (int j = 0; j < columns; j++)
        //    {
        //        sum += expCache[i, j];
        //    }

        //    for (int j = 0; j < columns; j++)
        //    {
        //        res[i, j] = expCache[i, j] / sum;
        //    }
        //}

        Debug.Assert(res.Cast<float>().All(x => !float.IsNaN(x)), "There should be no NaN values");

        return res;
    }

    // SoftMax function with log-sum-exp trick

    /// <summary>
    /// Applies the softmax function (with log-sum-exp trick) to the matrix.
    /// </summary>
    /// <returns>A new matrix with softmax-applied values.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] SoftmaxLogSumExp(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++) // rows = batch size (obervations)
        {
            float max = source[i, 0];
            for (int j = 1; j < columns; j++)
            {
                max = MathF.Max(max, source[i, j]);
            }

            float sum = 0;
            for (int j = 0; j < columns; j++)
            {
                sum += MathF.Exp(source[i, j] - max);
            }

            float logSumExp = max + MathF.Log(sum);

            for (int j = 0; j < columns; j++)
            {
                res[i, j] = source[i, j] - logSumExp;
            }
        }

        Debug.Assert(res.Cast<float>().All(x => !float.IsNaN(x)), "There should be no NaN values");

        return res;
    }

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise to the matrix.
    /// </summary>
    /// <returns>A new matrix with the hyperbolic tangent applied element-wise.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Tanh(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                res[i, j] = MathF.Tanh(source[i, j]);
            }
        }

        return res;
    }

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise to the matrix.
    /// </summary>
    /// <returns>A new matrix with the hyperbolic tangent applied element-wise.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,,,] Tanh(this float[,,,] source)
    {
        int dim1 = source.GetLength(0);
        int dim2 = source.GetLength(1);
        int dim3 = source.GetLength(2);
        int dim4 = source.GetLength(3);

        float[,,,] res = new float[dim1, dim2, dim3, dim4];

        for (int i = 0; i < dim1; i++)
        {
            for (int j = 0; j < dim2; j++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    for (int l = 0; l < dim4; l++)
                    {
                        res[i, j, k, l] = MathF.Tanh(source[i, j, k, l]);
                    }
                }
            }
        }

        return res;
    }

    /// <summary>
    /// Transposes the matrix by swapping its rows and columns.
    /// </summary>
    /// <returns>A new <see cref="float[,]"/> object representing the transposed matrix.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[,] Transpose(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);

        float[,] array = new float[columns, rows];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[j, i] = source[i, j];
            }
        }

        return array;
    }

    #endregion

    #region Shapes and Values

    public static bool HasSameShape(this float[] source, float[] matrix)
        => source.GetLength(0) == matrix.GetLength(0);

    public static bool HasSameShape(this float[,] source, float[,] matrix)
        => source.GetLength(0) == matrix.GetLength(0) && source.GetLength(1) == matrix.GetLength(1);

    public static bool HasSameShape(this float[,,,] source, float[,,,] matrix)
        => source.GetLength(0) == matrix.GetLength(0) && source.GetLength(1) == matrix.GetLength(1) && source.GetLength(2) == matrix.GetLength(2) && source.GetLength(3) == matrix.GetLength(3);

    public static bool HasSameValues(this float[,] source, float[,] matrix)
    {
        if (!source.HasSameShape(matrix))
        {
            return false;
        }

        for (int row = 0; row < source.GetLength(0); row++)
        {
            for (int col = 0; col < source.GetLength(1); col++)
            {
                if (source[row, col] != matrix[row, col])
                {
                    return false;
                }
            }
        }

        return true;
    }
    
    #endregion
}
