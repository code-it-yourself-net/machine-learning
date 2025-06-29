// Machine Learning Utils
// File name: Program5.cs
// Code It Yourself with .NET, 2024

internal class Program5
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
        // Each inner array represents a sample: [x1, x2, x3, y]
        // We are trying to find the relationship: y = 2*x1 + 3*x2 - 1*x3 + 5
        float[][] data = [
            [1, 2, 1, 12], // y = 2*1 + 3*2 - 1*1 + 5 = 12
            [2, 1, 2, 10], // y = 2*2 + 3*1 - 1*2 + 5 = 10
            [3, 3, 1, 19], // y = 2*3 + 3*3 - 1*1 + 5 = 19
            [4, 2, 3, 16], // y = 2*4 + 3*2 - 1*3 + 5 = 16
            [1, 4, 2, 17]  // y = 2*1 + 3*4 - 1*2 + 5 = 17
        ];

        // 3. Initialize model parameters
        // These are the coefficients for our independent variables and the bias term
        float[] a = new float[numCoefficients]; // Corresponds to a1, a2, a3. It's already initialized to 0 at this point.
        float b = 0;

        // 4. Training loop
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            // Initialize accumulators for errors and gradients for this iteration
            float sumSquaredError = 0;
            float[] sumErrorForWeights = new float[numCoefficients]; // Accumulator for each coefficient's gradient part
            float sumErrorForB = 0; // Accumulator for the bias's gradient part

            foreach (float[] sample in data)
            {
                // Separate independent variables (features) (x) from the dependent variable (target) (y)
                float[] x = new float[numCoefficients];
                for (int i = 0; i < numCoefficients; i++)
                {
                    x[i] = sample[i];
                }
                float y = sample[numCoefficients];

                // Prediction and error calculation
                // prediction = a1*x1 + a2*x2 + a3*x3 + b
                float prediction = b;
                for (int i = 0; i < numCoefficients; i++)
                {
                    prediction += a[i] * x[i];
                }
                float error = y - prediction;

                // Accumulate squared error for MSE calculation
                sumSquaredError += error * error;

                // Accumulate parts needed for gradient calculation
                // For each ai, the gradient part is (error * xi)
                for (int i = 0; i < numCoefficients; i++)
                {
                    sumErrorForWeights[i] += error * x[i];
                }
                // For the bias, the gradient part is just the error
                sumErrorForB += error;
            }

            // Number of samples
            int n = data.Length;

            // MSE (Mean Squared Error)
            float meanSquaredError = sumSquaredError / n;

            // Calculate gradients (partial derivatives of MSE)
            // ∂MSE/∂ai = -2/n * Σ(error * xi)
            float[] deltaA = new float[numCoefficients];
            for (int i = 0; i < numCoefficients; i++)
            {
                deltaA[i] = -2.0f / n * sumErrorForWeights[i];
            }

            // ∂MSE/∂b = -2/n * Σ(error)
            float deltaB = -2.0f / n * sumErrorForB;

            // Update regression parameters using gradient descent
            for (int i = 0; i < numCoefficients; i++)
            {
                a[i] -= lr * deltaA[i];
            }
            b -= lr * deltaB;

            if (iteration % printEvery == 0)
            {
                // Format the output string for coefficients
                string coefficientsStr = string.Join(", ", a.Select(val => $"{val,9:F4}"));
                Console.WriteLine($"Iteration: {iteration,6}, MSE: {meanSquaredError,10:F5}, coefficients: [{coefficientsStr}], b: {b,9:F4}");
            }
        }

        // 5. Output learned parameters
        Console.WriteLine("\n--- Training Complete ---");
        Console.WriteLine($"Learned parameters:");
        for (int i = 0; i < numCoefficients; i++)
        {
            Console.WriteLine($"  a{i + 1}: {a[i]:F4}");
        }
        Console.WriteLine($"  b (intercept):   {b:F4}");

        Console.WriteLine($"\nExpected parameters from the formula y = 2*x1 + 3*x2 - 1*x3 + 5:");
        Console.WriteLine($"  w1 =  2.0000, w2 =  3.0000, w3 = -1.0000, b = 5.0000");
        Console.ReadLine();
    }
}
