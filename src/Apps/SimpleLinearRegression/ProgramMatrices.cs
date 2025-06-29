// Machine Learning Utils
// File name: ProgramMatrices.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

internal class ProgramMatrices
{
    public static void Main()
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        // The model we are trying to learn is: y = a1*x1 + a2*x2 + a3*x3 + b

        // 1. Set the parameters for the model
        const float lr = 0.005f;   // Learning Rate
        const int iterations = 16_000;
        const int printEvery = 1_000;
        const int numCoefficients = 3; // Number of independent variables (a1, a2, a3)

        // 2. Prepare training data
        float[,] X = new float[,] {
            {1, 2, 1}, // Corresponds to x1, x2, x3 for the first sample
            {2, 1, 2}, // Corresponds to x1, x2, x3 for the second sample
            {3, 3, 1}, // and so on
            {4, 2, 3},
            {1, 4, 2}
        };

        float[,] Y = new float[,] {
            {12}, // Corresponds to the target value for the first sample
            {10}, // Corresponds to the target value for the second sample
            {19}, // and so on
            {16},
            {17}
        };

        // Number of samples
        int n = X.GetLength(0);

        // 3. Initialize model parameters
        // These are the coefficients for our independent variables and the bias term
        float[,] A = new float[numCoefficients, 1]; // Corresponds to a1, a2, a3. It's already initialized to 0 at this point.
        float b = 0;

        // 4. Training loop
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            // Make predictions for all samples at once: predictions = X * a + b
            float[,] predictions = X.MultiplyDot(A).Add(b);

            // Calculate errors for all samples: errors = Y - predictions
            float[,] errors = Y.Subtract(predictions);

            // Calculate the Mean Squared Error loss: MSE = mean(errors^2)
            float meanSquaredError = errors.Power(2).Mean();

            // Calculate gradient for coefficients 'a': ∂MSE/∂a = -2/n * X^T * errors
            // X.Transpose() aligns features with their corresponding errors for the dot product.
            float[,] deltaA = X.Transpose().MultiplyDot(errors).Multiply(-2.0f / n);

            // Calculate gradient for intercept 'b': ∂MSE/∂b = -2/n * sum(errors)
            float deltaB = errors.Sum() * (-2.0f / n);

            // Update regression parameters using gradient descent
            // a = a - learningRate * deltaA
            A = A.Subtract(deltaA.Multiply(lr));

            // b = b - learningRate * deltaB
            b -= lr * deltaB;

            if (iteration % printEvery == 0)
            {
                // Format the output string for coefficients
                // string coefficientsStr = string.Join(", ", A.Select(val => $"{val,9:F4}"));
                string coefficientsStr = string.Join(", ", A.Cast<float>().Select(val => $"{val,9:F4}"));
                Console.WriteLine($"Iteration: {iteration,6}, MSE: {meanSquaredError,10:F5}, coefficients: [{coefficientsStr}], b: {b,9:F4}");
            }
        }

        // 5. Output learned parameters
        Console.WriteLine("\n--- Training Complete ---");
        Console.WriteLine($"Learned parameters:");
        for (int i = 0; i < numCoefficients; i++)
        {
            Console.WriteLine($"  a{i + 1}: {A[i, 0]:F4}");
        }
        Console.WriteLine($"  b (intercept):   {b:F4}");

        Console.WriteLine($"\nExpected parameters from the formula y = 2*x1 + 3*x2 - 1*x3 + 5:");
        Console.WriteLine($"  a1 =  2.0000, a2 =  3.0000, a3 = -1.0000, b = 5.0000");
        Console.ReadLine();
    }
}

public static class ArrayExtensions
{
    public static float[,] MultiplyDot(this float[,] source, float[,] matrix)
    {

        Debug.Assert(source.GetLength(1) == matrix.GetLength(0));

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

    public static float[,] Subtract(this float[,] source, float[,] matrix)
    {
        Debug.Assert(source.GetLength(0) == matrix.GetLength(0));
        Debug.Assert(source.GetLength(1) == matrix.GetLength(1));

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

    public static float Mean(this float[,] source)
        => source.Sum() / source.Length;

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
}
