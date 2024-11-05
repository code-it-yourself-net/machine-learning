// Machine Learning Utils
// File name: Matrix.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning;

/// <summary>
/// Represents a matrix of floating-point numbers.
/// </summary>
public class Matrix
{

#if DEBUG
    private const string NumberOfColumnsMustBeEqualToNumberOfColumnsMsg = "The number of columns of the first matrix must be equal to the number of columns of the second matrix.";
    private const string NumberOfRowsMustBeEqualToNumberOfRowsMsg = "The number of rows of the first matrix must be equal to the number of rows of the second matrix.";
    private const string NumberOfColumnsMustBeEqualToNumberOfRowsMsg = "The number of columns of the first matrix must be equal to the number of rows of the second matrix.";
    private const string NumberOfRowsMustBeEqualToOneMsg = "The number of rows of the second matrix must be equal to 1.";
    private const string InvalidSizesMsg = "The sizes of the matrices are not compatible for elementwise multiplication.";
#endif

    private readonly float[,] _array;

    /// <summary>
    /// Initializes a new instance of the <see cref="Matrix"/> class with the specified array.
    /// </summary>
    /// <param name="array">The array representing the matrix.</param>
    /// <remarks>
    /// A new instance of the <see cref="Matrix"/> class is filled with zeros.
    /// </remarks>
    public Matrix(float[,] array)
    {
        _array = array;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Matrix"/> class with the specified number of rows and columns.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <remarks>
    /// A new instance of the <see cref="Matrix"/> class is filled with zeros.
    /// </remarks>
    public Matrix(int rows, int columns)
    {
        _array = new float[rows, columns];
    }

    public float[,] Array => _array;

    /// <summary>
    /// Implicitly converts a <see cref="Matrix"/> to an <see cref="Array"/>.
    /// </summary>
    /// <param name="matrix">The <see cref="Matrix"/> to convert.</param>
    /// <returns>The converted <see cref="Array"/>.</returns>
    public static explicit operator float[,](Matrix matrix) => matrix.Array;

    #region Zeros, Ones, and Random

    /// <summary>
    /// Creates a new matrix filled with zeros, with the same dimensions as the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix used to determine the dimensions of the new matrix.</param>
    /// <returns>A new matrix filled with zeros.</returns>
    public static Matrix Zeros(Matrix matrix)
    {
        (int rows, int columns) = GetDimensions(matrix);
        return Zeros(rows, columns);
    }

    /// <summary>
    /// Creates a new matrix filled with zeros, with the specified number of rows and columns.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <returns>A new matrix filled with zeros.</returns>
    public static Matrix Zeros(int rows, int columns) => new(rows, columns);

    public static Matrix ZeroOnes(Matrix matrix, float onesProbability, Random random)
    {
        (int rows, int columns) = GetDimensions(matrix);
        float[,] array = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                if(random.NextDouble() < onesProbability)
                {
                    array[i, j] = 1;
                }
            }
        }
        return new Matrix(array);
    }

    /// <summary>
    /// Creates a new matrix filled with ones, with the same dimensions as the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix used to determine the dimensions of the new matrix.</param>
    /// <returns>A new matrix filled with ones.</returns>
    public static Matrix Ones(Matrix matrix)
    {
        (int rows, int columns) = GetDimensions(matrix);
        return Ones(rows, columns);
    }

    /// <summary>
    /// Creates a new matrix filled with ones, with the specified number of rows and columns.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <returns>A new matrix filled with ones.</returns>
    public static Matrix Ones(int rows, int columns)
    {
        // Create an instance of Array of floats using rows and columns and fill it with ones.
        float[,] array = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = 1f;
            }
        }
        return new Matrix(array);
    }

    /// <summary>
    /// Creates a new matrix filled with random values between -0.5 and 0.5, with the specified number of rows and columns.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="columns">The number of columns in the matrix.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>A new matrix filled with random values.</returns>
    public static Matrix Random(int rows, int columns, Random random)
    {
        // Create an instance of Array of floats using rows and columns and fill it with randoms.
        float[,] array = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = random.NextSingle() - 0.5f;
            }
        }
        return new Matrix(array);
    }

    public static Matrix RandomNormal(int rows, int columns, Random random, float mean = 0, float stdDev = 1) 
    {
        float[,] array = new float[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = BoxMuller() * stdDev + mean;
            }
        }
        return new Matrix(array);

        float BoxMuller() // TODO: Move to RandomUtils
        {
            // uniform(0,1] random doubles
            double u1 = random.NextDouble(); 
            double u2 = random.NextDouble();

            //random normal(0,1)
            float randStdNormal = Convert.ToSingle(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2)); 
            return randStdNormal;
        }
    }

    public static Matrix Range(int rows, int columns, float from, float to)
    {
        float[,] array = new float[rows, columns];
        // float step = (to - from) / (rows * columns);
        float step = (to - from) / columns;
        for (int i = 0; i < rows; i++)
        {
            float value = from;
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = value;
                value += step;
            }
        }
        return new Matrix(array);
    }

    #endregion

    #region Operations with scalar

    /// <summary>
    /// Adds a scalar value to each element of the matrix.
    /// </summary>
    /// <param name="scalar">The scalar value to add.</param>
    /// <returns>A new matrix with the scalar added to each element.</returns>
    public Matrix Add(float scalar)
    {
        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = _array[i, j] + scalar;
            }
        }

        return new Matrix(array);
    }

    public void AddInPlace(float scalar)
    {
        for (int row = 0; row < _array.GetLength(0); row++)
        {
            for (int column = 0; column < _array.GetLength(1); column++)
            {
                _array[row, column] += scalar;
            }
        }
    }

    public void DivideInPlace(float scalar)
    {
        for (int row = 0; row < _array.GetLength(0); row++)
        {
            for (int column = 0; column < _array.GetLength(1); column++)
            {
                _array[row, column] /= scalar;
            }
        }
    }

    /// <summary>
    /// Multiplies each element of the matrix by a scalar value.
    /// </summary>
    /// <param name="scalar">The scalar value to multiply.</param>
    /// <returns>A new matrix with each element multiplied by the scalar value.</returns>
    public Matrix Multiply(float scalar)
    {
        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = _array[i, j] * scalar;
            }
        }

        return new Matrix(array);
    }

    public void MultiplyInPlace(float scalar) 
    { 
        for (int row = 0; row < _array.GetLength(0); row++)
        {
            for (int column = 0; column < _array.GetLength(1); column++)
            {
                _array[row, column] *= scalar;
            }
        }
    }

    public Matrix Divide(float scalar)
    {
        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = _array[i, j] / scalar;
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Raises each element of the matrix to the specified power.
    /// </summary>
    /// <param name="scalar">The power to raise each element to.</param>
    /// <returns>A new matrix with each element raised to the specified power.</returns>
    public Matrix Power(int scalar)
    {
        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = MathF.Pow(_array[i, j], scalar);
            }
        }

        return new Matrix(array);
    }

    #endregion

    #region Operations with matrix

    /// <summary>
    /// Adds the elements of the specified matrix to the current matrix.
    /// </summary>
    /// <param name="matrix">The matrix to add.</param>
    /// <returns>A new matrix with the elements added.</returns>
    /// <exception cref="Exception">Thrown when the number of rows in the specified matrix is not equal to the number of rows in the current matrix, or when the number of columns in the specified matrix is not equal to the number of columns in the current matrix.</exception>
    public Matrix Add(Matrix matrix)
    {

#if DEBUG
        if (GetDimension(Dimension.Rows) != matrix.GetDimension(Dimension.Rows))
            throw new Exception(NumberOfRowsMustBeEqualToNumberOfRowsMsg);

        if (GetDimension(Dimension.Columns) != matrix.GetDimension(Dimension.Columns))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);
#endif

        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();
        float[,] matrixArray = matrix.Array;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = _array[i, j] + matrixArray[i, j];
            }
        }

        return new Matrix(array);
    }

    public void AddInPlace(Matrix matrix) 
    {

#if DEBUG
        if (GetDimension(Dimension.Rows) != matrix.GetDimension(Dimension.Rows))
            throw new Exception(NumberOfRowsMustBeEqualToNumberOfRowsMsg);

        if (GetDimension(Dimension.Columns) != matrix.GetDimension(Dimension.Columns))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);
#endif

        float[,] matrixArray = matrix.Array;
        for (int row = 0; row < _array.GetLength(0); row++)
        {
            for (int column = 0; column < _array.GetLength(1); column++)
            {
                _array[row, column] += matrixArray[row, column];
            }
        }
    }

    /// <summary>
    /// Adds a row to the current matrix by elementwise addition with the specified matrix.
    /// </summary>
    /// <param name="row">The matrix to add as a row.</param>
    /// <returns>A new matrix with the row added.</returns>
    /// <exception cref="Exception">Thrown when the number of columns in the specified matrix is not equal to the number of columns in the current matrix, or when the number of rows of the specified matrix is not equal to 1.</exception>
    public Matrix AddRow(Matrix row)
    {

#if DEBUG
        if (GetDimension(Dimension.Columns) != row.GetDimension(Dimension.Columns))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);

        if (row.GetDimension(Dimension.Rows) != 1)
            throw new Exception(NumberOfRowsMustBeEqualToOneMsg);
#endif

        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);
        float[,] rowArray = row.Array;

        float[,] array = new float[rows, columns]; 

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = _array[i, j] + rowArray[0, j];
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Clips the values of the matrix in-place between the specified minimum and maximum values.
    /// </summary>
    /// <param name="min">The minimum value to clip the matrix elements to.</param>
    /// <param name="max">The maximum value to clip the matrix elements to.</param>
    public void ClipInPlace(float min, float max)
    {
        for (int i = 0; i < _array.GetLength(0); i++)
        {
            for (int j = 0; j < _array.GetLength(1); j++)
            {
                _array[i, j] = MathF.Max(min, MathF.Min(max, _array[i, j]));
            }
        }
    }

    /// <summary>
    /// Multiplies the current matrix with another matrix using the dot product.
    /// </summary>
    /// <param name="matrix">The matrix to multiply with.</param>
    /// <returns>A new matrix that is the result of the dot product multiplication.</returns>
    /// <exception cref="Exception">Thrown when the number of columns in the current matrix is not equal to the number of rows in the specified matrix.</exception>
    public Matrix MultiplyDot(Matrix matrix)
    {

#if DEBUG
        if (GetDimension(Dimension.Columns) != matrix.GetDimension(Dimension.Rows))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfRowsMsg);
#endif

        int matrixColumns = matrix.Array.GetLength(1);

        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);

        float[,] array = new float[rows, matrixColumns];
        float[,] matrixArray = matrix.Array;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < matrixColumns; j++)
            {
                float sum = 0;
                for (int k = 0; k < columns; k++)
                {
                    sum += _array[i, k] * matrixArray[k, j];
                }
                array[i, j] = sum;
            }
        }

        return new Matrix(array);
    }

//    public Matrix MultiplyDotWithMatrixArray(Matrix matrix)
//    {

//#if DEBUG
//        if (GetDimension(Dimension.Columns) != matrix.GetDimension(Dimension.Rows))
//            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfRowsMsg);
//#endif

//        int matrixColumns = matrix.Array.GetLength(1);

//        int rows = _array.GetLength(0);
//        int columns = _array.GetLength(1);

//        float[,] array = new float[rows, matrixColumns];
//        float[,] matrixArray = matrix.Array;

//        for (int i = 0; i < rows; i++)
//        {
//            for (int j = 0; j < matrixColumns; j++)
//            {
//                float sum = 0;
//                for (int k = 0; k < columns; k++)
//                {
//                    sum += _array[i, k] * matrixArray[k, j];
//                }
//                array[i, j] = sum;
//            }
//        }

//        return new Matrix(array);
//    }

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
    public Matrix MultiplyElementwise(Matrix matrix)
    {
        int thisRows = _array.GetLength(0);
        int thisColumns = _array.GetLength(1);
        int matrixRows = matrix.GetDimension(Dimension.Rows);
        int matrixColumns = matrix.GetDimension(Dimension.Columns);

        int maxRows = Math.Max(thisRows, matrixRows);
        int maxColumns = Math.Max(thisColumns, matrixColumns);

#if DEBUG
        // Make sure that the analogous sizes of both matrices are multiples of each other or - especially - are equal
        if (maxRows % thisRows != 0 || maxRows % matrixRows != 0 || maxColumns % thisColumns != 0 || maxColumns % matrixColumns != 0)
        {
            throw new Exception(InvalidSizesMsg);
        }
#endif

        float[,] array = new float[maxRows, maxColumns];
        float[,] matrixArray = matrix.Array;

        for (int i = 0; i < maxRows; i++)
        {
            for (int j = 0; j < maxColumns; j++)
            {
                float thisValue = _array[i % thisRows, j % thisColumns];
                float matrixValue = matrixArray[i % matrixRows, j % matrixColumns];
                array[i, j] = thisValue * matrixValue;
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Subtracts the elements of the specified matrix from the current matrix.
    /// </summary>
    /// <param name="matrix">The matrix to subtract.</param>
    /// <returns>A new matrix with the elements subtracted.</returns>
    /// <exception cref="Exception">Thrown when the number of rows in the specified matrix is not equal to the number of rows in the current matrix, or when the number of columns in the specified matrix is not equal to the number of columns in the current matrix.</exception>
    public Matrix Subtract(Matrix matrix)
    {
#if DEBUG
        if (GetDimension(Dimension.Rows) != matrix.GetDimension(Dimension.Rows))
            throw new Exception(NumberOfRowsMustBeEqualToNumberOfRowsMsg);

        if (GetDimension(Dimension.Columns) != matrix.GetDimension(Dimension.Columns))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);
#endif

        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();
        float[,] matrixArray = matrix.Array;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = _array[i, j] - matrixArray[i, j];
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Subtracts the elements of the specified matrix from this matrix in-place.
    /// </summary>
    /// <param name="matrix">The matrix to subtract.</param>
    /// <exception cref="Exception">
    /// Thrown when the number of rows in the specified matrix is not equal to the number of rows in this matrix,
    /// or when the number of columns in the specified matrix is not equal to the number of columns in this matrix.
    /// </exception>
    public void SubtractInPlace(Matrix matrix)
    {

#if DEBUG
        if (GetDimension(Dimension.Rows) != matrix.GetDimension(Dimension.Rows))
            throw new Exception(NumberOfRowsMustBeEqualToNumberOfRowsMsg);

        if (GetDimension(Dimension.Columns) != matrix.GetDimension(Dimension.Columns))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);
#endif

        float[,] matrixArray = matrix.Array;

        for (int i = 0; i < _array.GetLength(0); i++)
        {
            for (int j = 0; j < _array.GetLength(1); j++)
            {
                _array[i, j] -= matrixArray[i, j];
            }
        }
    }

    #endregion

    #region Aggregations

    public float Max()
    {
        float max = float.MinValue;
        
        for (int i = 0; i < _array.GetLength(0); i++)
        {
            for (int j = 0; j < _array.GetLength(1); j++)
            {
                max = Math.Max(max, _array[i, j]);
            }
        }
        return max;
    }

    /// <summary>
    /// Calculates the mean of all elements in the matrix.
    /// </summary>
    /// <returns>The mean of all elements in the matrix.</returns>
    public float Mean() => Sum() / _array.Length;

    public float Min()
    {
        float max = float.MaxValue;

        for (int i = 0; i < _array.GetLength(0); i++)
        {
            for (int j = 0; j < _array.GetLength(1); j++)
            {
                max = Math.Min(max, _array[i, j]);
            }
        }
        return max;
    }

    /// <summary>
    /// Calculates the standard deviation.
    /// </summary>
    /// <returns></returns>
    public float Std()
    {
        float mean = Mean();
        float sum = 0;
        for (int i = 0; i < _array.GetLength(0); i++)
        {
            for (int j = 0; j < _array.GetLength(1); j++)
            {
                sum += MathF.Pow(_array[i, j] - mean, 2);
            }
        }

        return (float)Math.Sqrt(sum / _array.Length);
    }

    /// <summary>
    /// Calculates the sum of all elements in the matrix.
    /// </summary>
    /// <returns>The sum of all elements in the matrix.</returns>
    public float Sum()
    {
        // Sum over all elements.
        float sum = 0;

        for (int i = 0; i < _array.GetLength(0); i++)
        {
            for (int j = 0; j < _array.GetLength(1); j++)
            {
                sum += _array[i, j];
            }
        }
        
        return sum;
    }

    /// <summary>
    /// Calculates the sum of the matrix elements along the specified dimension.
    /// </summary>
    /// <param name="dimension">The dimension along which to calculate the sum.</param>
    /// <returns>A new Matrix object containing the sum of the elements along the specified dimension.</returns>
    public Matrix SumBy(Dimension dimension)
    {
        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);

        float[,] array = new float[1, dimension == Dimension.Rows ? columns : rows];

        for (int i = 0; i < (dimension == Dimension.Rows ? columns : rows); i++)
        {
            float sum = 0;
            for (int j = 0; j < (dimension == Dimension.Rows ? rows : columns); j++)
            {
                sum += _array[dimension == Dimension.Rows ? j : i, dimension == Dimension.Rows ? i : j];
            }
            array[0, i] = sum;
        }

        return new Matrix(array);
    }

    public Matrix AvgBy(Dimension dimension)
    {
        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);

        float[,] array = new float[1, dimension == Dimension.Rows ? columns : rows];

        for (int i = 0; i < (dimension == Dimension.Rows ? columns : rows); i++)
        {
            float sum = 0;
            for (int j = 0; j < (dimension == Dimension.Rows ? rows : columns); j++)
            {
                sum += _array[dimension == Dimension.Rows ? j : i, dimension == Dimension.Rows ? i : j];
            }
            array[0, i] = sum / (dimension == Dimension.Rows ? rows : columns);
        }

        return new Matrix(array);
    }

    #endregion

    #region Slices, Rows, and Columns

    /// <summary>
    /// Gets a row from the matrix.
    /// </summary>
    /// <param name="row">The index of the row to retrieve.</param>
    /// <returns>A new <see cref="Matrix"/> object representing the specified row.</returns>
    /// <remarks>
    /// The returned row is a new instance of the <see cref="Matrix"/> class and has the same number of columns as the original matrix.
    /// </remarks>
    public Matrix GetRow(int row)
    {
        int columns = _array.GetLength(1);

        // Create an array to store the row.
        float[,] newArray = new float[1, columns];
        for (int i = 0; i < columns; i++)
        {
            // Access each element in the specified row.
            newArray[0, i] = _array[row, i];
        }

        return new Matrix(newArray);
    }

    /// <summary>
    /// Sets the values of a specific row in the matrix.
    /// </summary>
    /// <param name="row">The index of the row to set.</param>
    /// <param name="matrix">The matrix containing the values to set.</param>
    /// <exception cref="Exception">Thrown when the number of columns in the specified matrix is not equal to the number of columns in the current matrix.</exception>
    public void SetRow(int row, Matrix matrix)
    {

#if DEBUG
        if (matrix.GetDimension(Dimension.Columns) != _array.GetLength(1))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);

        if(matrix.GetDimension(Dimension.Rows) != 1)
            throw new Exception(NumberOfRowsMustBeEqualToOneMsg);
#endif

        float[,] matrixArray = matrix.Array;

        for (int i = 0; i < _array.GetLength(1); i++)
        {
            _array[row, i] = matrixArray[0, i];
        }
    }

    /// <summary>
    /// Gets a submatrix containing the specified range of rows from the current matrix.
    /// </summary>
    /// <param name="range">The range of rows to retrieve.</param>
    /// <returns>A new <see cref="Matrix"/> object representing the submatrix.</returns>
    /// <remarks>
    /// The returned rows are a new instance of the <see cref="Matrix"/> class and have the same number of columns as the original matrix.
    /// </remarks>
    public Matrix GetRows(Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(_array.GetLength(0));

        int columns = _array.GetLength(1);
        float[,] newArray = new float[length, columns];

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                newArray[i, j] = _array[i + offset, j];
            }
        }

        return new Matrix(newArray);
    }

    /// <summary>
    /// Gets a submatrix containing the specified column from the current matrix. The shape is [rows, 1].
    /// </summary>
    /// <param name="column"></param>
    /// <returns></returns>
    public Matrix GetColumn(int column)
    {
        int rows = _array.GetLength(0);

        // Create an array to store the column.
        float[,] newArray = new float[rows, 1];

        for (int i = 0; i < rows; i++)
        {
            // Access each element in the specified column.
            newArray[i, 0] = _array[i, column];
        }

        return new Matrix(newArray);
    }

    /// <summary>
    /// Gets a submatrix containing the specified range of columns from the current matrix. The shape is [rows, range].
    /// </summary>
    /// <param name="column"></param>
    /// <returns></returns>
    public Matrix GetColumns(Range range)
    {
        (int offset, int length) = range.GetOffsetAndLength(_array.GetLength(1));

        int rows = _array.GetLength(0);
        float[,] newArray = new float[rows, length];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < length; j++)
            {
                newArray[i, j] = _array[i, j + offset];
            }
        }

        return new Matrix(newArray);
    }

    #endregion

    #region Matrix operations and functions

    public Matrix Argmax()
    {
        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);

        float[,] array = new float[rows, 1];

        for (int i = 0; i < rows; i++)
        {
            float max = float.MinValue;
            int maxIndex = 0;
            for (int j = 0; j < columns; j++)
            {
                float value = _array[i, j];
                if (value > max)
                {
                    max = value;
                    maxIndex = j;
                }
            }
            array[i, 0] = maxIndex;
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Compares two matrices element-wise and returns a new matrix with 1s where the elements are equal and 0s where they are not.
    /// </summary>
    /// <param name="matrix">The matrix to compare with.</param>
    /// <returns>A new matrix with 1s where the elements are equal and 0s where they are not.</returns>
    public Matrix Compare(Matrix matrix)
    {

#if DEBUG
        if (GetDimension(Dimension.Rows) != matrix.GetDimension(Dimension.Rows))
            throw new Exception(NumberOfRowsMustBeEqualToNumberOfRowsMsg);

        if (GetDimension(Dimension.Columns) != matrix.GetDimension(Dimension.Columns))
            throw new Exception(NumberOfColumnsMustBeEqualToNumberOfColumnsMsg);
#endif

        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();
        float[,] matrixArray = matrix.Array;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = _array[i, j].Equals(matrixArray[i, j]) ? 1 : 0;
            }
        }

        return new Matrix(array);
    }

    public Matrix Log()
    {
        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = MathF.Log(_array[i, j]);
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Applies the sigmoid function to each element of the matrix.
    /// </summary>
    /// <returns>A new matrix with each element transformed by the sigmoid function with the same dimensions as the original matrix.</returns>
    public Matrix Sigmoid()
    {
        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i , j] = 1 / (1 + MathF.Exp(-_array[i, j]));
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Calculates the derivative of the sigmoid function for each element of the matrix.
    /// </summary>
    /// <remarks>
    /// The derivative of the sigmoid function is calculated as: sigmoid(x) * (1 - sigmoid(x)).
    /// </remarks>
    /// <returns>A new matrix with each element transformed by the derivative of the sigmoid function with the same dimensions as the original matrix.</returns>
    public Matrix SigmoidDerivative()
    {
        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                float sigmoid = 1 / (1 + MathF.Exp(-_array[i, j]));
                array[i, j] = sigmoid * (1 - sigmoid);
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Applies the softmax function to the matrix.
    /// </summary>
    /// <returns>A new matrix with softmax-applied values.</returns>
    /// <remarks>Softmax formula: <c>exp(x) / sum(exp(x))</c>.</remarks>
    public Matrix Softmax()
    {
        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();

        float[,] expCache = new float[rows, columns];
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < columns; j++)
            {
                expCache[i, j] = MathF.Exp(_array[i, j]);
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
                array[i, j] = expCache[i, j] / sum;
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise to the matrix.
    /// </summary>
    /// <returns>A new matrix with the hyperbolic tangent applied element-wise.</returns>
    public Matrix Tanh()
    {
        (float[,] array, int rows, int columns) = CreateEmptyCopyAsArray();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = MathF.Tanh(_array[i, j]);
            }
        }

        return new Matrix(array);
    }

    /// <summary>
    /// Transposes the matrix by swapping its rows and columns.
    /// </summary>
    /// <returns>A new <see cref="Matrix"/> object representing the transposed matrix.</returns>
    public Matrix Transpose()
    {
        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);

        float[,] array = new float[columns, rows];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[j, i] = _array[i, j];
            }
        }

        return new Matrix(array);
    }

    #endregion

    /// <summary>
    /// Gets the number of rows or columns in the matrix.
    /// </summary>
    /// <param name="dimension">The dimension to get the size of.</param>
    public int GetDimension(Dimension dimension) 
        => _array.GetLength((int)dimension);

    private static (int Rows, int Columns) GetDimensions(Matrix matrix) 
        => (matrix.GetDimension(Dimension.Rows), matrix.GetDimension(Dimension.Columns));

    /// <summary>
    /// Creates a new empty instance of the <see cref="System.Array"/> class with the same dimensions as this matrix.
    /// </summary>
    /// <returns>A tuple containing the new array filled with zeros, the number of rows, and the number of columns.</returns>
    private (float[,] Array, int Rows, int Columns) CreateEmptyCopyAsArray()
    {
        int rows = _array.GetLength(0);
        int columns = _array.GetLength(1);
        float[,] array = new float[rows, columns];
        return (array, rows, columns);
    }

    public override string ToString()
    {
        string result = string.Empty;
        int rows = _array.GetLength(0);
        int cols = _array.GetLength(1);
        for (int i = 0; i < Math.Min(rows, 5); i++)
        {
            string rowResult = string.Empty;
            for (int j = 0; j < Math.Min(cols, 5); j++)
            {
                rowResult += $"{_array.GetValue(i, j)}";
                if (j < cols - 1)
                {
                    rowResult += "; ";
                }
            }
            result += $"[{rowResult}]";

            if (i < rows - 1)
            {
                result += ", \n";
            }
        }
        return $"({rows}*{cols}): \n{result}";
    }

    /// <summary>
    /// Determines whether the specified matrix has the same shape as the current matrix.
    /// </summary>
    /// <param name="matrix">The matrix to compare.</param>
    /// <returns><c>true</c> if the specified matrix has the same shape as the current matrix; otherwise, <c>false</c>.</returns>
    public bool HasSameShape(Matrix matrix) => GetDimension(Dimension.Rows) == matrix.GetDimension(Dimension.Rows)
        && GetDimension(Dimension.Columns) == matrix.GetDimension(Dimension.Columns);

    public bool HasSameValues(Matrix matrix) {
        if (!HasSameShape(matrix))
        {
            return false;
        }

        for (int i = 0; i < _array.GetLength(0); i++)
        {
            for (int j = 0; j < _array.GetLength(1); j++)
            {
                if ((float)_array.GetValue(i, j)! != (float)matrix.Array.GetValue(i, j)!)
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>
    /// Clones the matrix.
    /// </summary>
    /// <returns>A copy of the matrix.</returns>
    public Matrix Clone() => new((float[,])_array.Clone());

    public static Matrix LoadCsv(string filePath)
    {
        string[] lines = File.ReadAllLines(filePath);
        int rows = lines.Length;
        int cols = lines[0].Split(',').Length;
        Matrix matrix = new(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            string[] values = lines[i].Split(',');
            for (int j = 0; j < cols; j++)
            {
                matrix.Array.SetValue(float.Parse(values[j]), i, j);
            }
        }
        return matrix;
    }

    
}