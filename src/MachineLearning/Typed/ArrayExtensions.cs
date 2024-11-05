// Machine Learning Utils
// File name: ArrayExtensions.cs
// Code It Yourself with .NET, 2024

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
    /// <param name="matrix">The matrix used to determine the dimensions of the new matrix.</param>
    /// <returns>A new matrix filled with ones.</returns>
    public static float[,] AsOnes(this float[,] source)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < columns; col++)
            {
                res[row, col] = 1;
            }
        }

        return res;
    }

    /// <summary>
    /// Creates a new matrix filled with ones, with the same dimensions as the specified matrix.
    /// </summary>
    /// <param name="source">The matrix used to determine the dimensions of the new matrix.</param>
    /// <returns>A new matrix filled with ones.</returns>
    public static float[] AsOnes(this float[] source)
    {
        int columns = source.GetLength(0);
        float[] res = new float[columns];

        for (int col = 0; col < columns; col++)
        {
            res[col] = 1;
        }

        return res;
    }

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

    #endregion

    #region Slices, Rows, and Columns

    /// <summary>
    /// Gets a submatrix containing the specified column from the current matrix. The shape is [rows, 1].
    /// </summary>
    /// <param name="column"></param>
    /// <returns></returns>
    public static float[,] GetColumn(this float[,] source, int column)
    {
        int rows = source.GetLength((int)Dimension.Rows);

        // Create an array to store the column.
        float[,] res = new float[rows, 1];

        for (int row = 0; row < rows; row++)
        {
            // Access each element in the specified column.
            res[row, 0] = source[row, column];
        }

        return res;
    }

    /// <summary>
    /// Gets a submatrix containing the specified range of columns from the current matrix. The shape is [rows, range].
    /// </summary>
    /// <returns></returns>
    public static float[,] GetColumns(this float[,] source, Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(source.GetLength((int)Dimension.Columns));

        int rows = source.GetLength((int)Dimension.Rows);
        float[,] res = new float[rows, length];

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < length; col++)
            {
                res[row, col] = source[row, col + offset];
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

    /// <summary>
    /// Gets a submatrix containing the specified range of rows from the current matrix.
    /// </summary>
    /// <param name="range">The range of rows to retrieve.</param>
    /// <returns>A new <see cref="Matrix"/> object representing the submatrix.</returns>
    /// <remarks>
    /// The returned rows are a new instance of the <see cref="Matrix"/> class and have the same number of columns as the original matrix.
    /// </remarks>
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

    /// <summary>
    /// Sets the values of a specific row in the matrix.
    /// </summary>
    /// <param name="rowIndex">The index of the row to set.</param>
    /// <param name="row">The matrix containing the values to set.</param>
    /// <exception cref="Exception">Thrown when the number of columns in the specified matrix is not equal to the number of columns in the current matrix.</exception>
    public static void SetRow(this float[,] source, int rowIndex, float[] row)
    {

#if DEBUG
        if (row.GetLength(0) != source.GetLength(1))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);
#endif

        for (int col = 0; col < source.GetLength(1); col++)
        {
            source[rowIndex, col] = row[col];
        }
    }

    #endregion

    #region Aggregations

    public static float Max(this float[,] source)
    {
        float max = float.MinValue;

        for (int row = 0; row < source.GetLength(0); row++)
        {
            for (int col = 0; col < source.GetLength(1); col++)
            {
                max = Math.Max(max, source[row, col]);
            }
        }
        return max;
    }

    /// <summary>
    /// Calculates the mean of all elements in the matrix.
    /// </summary>
    /// <returns>The mean of all elements in the matrix.</returns>
    public static float Mean(this float[,] source) => source.Sum() / source.Length;

    public static float Min(this float[,] source)
    {
        float min = float.MaxValue;

        for (int row = 0; row < source.GetLength(0); row++)
        {
            for (int col = 0; col < source.GetLength(1); col++)
            {
                min = Math.Min(min, source[row, col]);
            }
        }
        return min;
    }

    /// <summary>
    /// Calculates the standard deviation.
    /// </summary>
    /// <returns></returns>
    public static float Std(this float[,] source)
    {
        float mean = source.Mean();
        float sum = 0;
        for (int row = 0; row < source.GetLength(0); row++)
        {
            for (int col = 0; col < source.GetLength(1); col++)
            {
                sum += MathF.Pow(source[row, col] - mean, 2);
            }
        }

        return (float)Math.Sqrt(sum / source.Length);
    }

    /// <summary>
    /// Calculates the sum of all elements in the matrix.
    /// </summary>
    /// <returns>The sum of all elements in the matrix.</returns>
    public static float Sum(this float[,] source)
    {
        // Sum over all elements.
        float sum = 0;

        for (int row = 0; row < source.GetLength(0); row++)
        {
            for (int col = 0; col < source.GetLength(1); col++)
            {
                sum += source[row, col];
            }
        }

        return sum;
    }

    /// <summary>
    /// Summary of rows in one row.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
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

    public static void AddInPlace(this float[,] source, float scalar)
    {
        for (int row = 0; row < source.GetLength(0); row++)
        {
            for (int col = 0; col < source.GetLength(1); col++)
            {
                source[row, col] += scalar;
            }
        }
    }

    public static void DivideInPlace(this float[,] source, float scalar)
    {
        for (int row = 0; row < source.GetLength(0); row++)
        {
            for (int col = 0; col < source.GetLength(1); col++)
            {
                source[row, col] /= scalar;
            }
        }
    }

    /// <summary>
    /// Multiplies each element of the matrix by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply.</param>
    /// <returns>A new matrix with each element multiplied by the scalar value.</returns>
    public static float[,] Multiply(this float[,] source, float scalar)
    {
        int rows = source.GetLength(0);
        int columns = source.GetLength(1);
        float[,] res = new float[rows, columns];

        for (int col = 0; col < rows; col++)
        {
            for (int row = 0; row < columns; row++)
            {
                res[col, row] = source[col, row] * scalar;
            }
        }

        return res;
    }

    public static void MultiplyInPlace(this float[,] source, float scalar)
    {
        for (int row = 0; row < source.GetLength(0); row++)
        {
            for (int column = 0; column < source.GetLength(1); column++)
            {
                source[row, column] *= scalar;
            }
        }
    }

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
    public static void ClipInPlace(this float[,] source, float min, float max)
    {
        for (int row = 0; row < source.GetLength(0); row++)
        {
            for (int col = 0; col < source.GetLength(1); col++)
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

    #endregion

    #region Matrix operations and functions

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
        }

        for (int i = 0; i < rows; i++)
        {
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

        return res;
    }

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise to the matrix.
    /// </summary>
    /// <returns>A new matrix with the hyperbolic tangent applied element-wise.</returns>
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
    /// Transposes the matrix by swapping its rows and columns.
    /// </summary>
    /// <returns>A new <see cref="float[,]"/> object representing the transposed matrix.</returns>
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

    public static bool HasSameShape(this float[,] source, float[,] matrix)
        => source.GetLength(0) == matrix.GetLength(0) && source.GetLength(1) == matrix.GetLength(1);

    public static bool HasSameShape(this float[] source, float[] matrix)
        => source.GetLength(0) == matrix.GetLength(0);

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
