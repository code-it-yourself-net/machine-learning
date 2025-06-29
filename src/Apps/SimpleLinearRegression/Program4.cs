// Machine Learning Utils
// File name: ProgramSimple.cs
// Code It Yourself with .NET, 2024

internal class Program4
{
    public static void Main()
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        // The model we are trying to learn is: y = a1*x1 + a2*x2 + a3*x3 + b

        // 1. Set the parameters for the model
        const float lr = 0.005f;   // Learning Rate
        const int iterations = 16_000;
        const int printEvery = 1_000;

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
        float a1 = 0, a2 = 0, a3 = 0; // Parameters for x1, x2, x3
        float b = 0;

        // 4. Training loop
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            // Initialize accumulators for errors and gradients for this iteration
            float sumSquaredError = 0;
            float sumErrorForA1 = 0; // Accumulator for a1's gradient part
            float sumErrorForA2 = 0; // Accumulator for a2's gradient part
            float sumErrorForA3 = 0; // Accumulator for a3's gradient part
            float sumErrorForB = 0;  // Accumulator for the bias's gradient part

            foreach (float[] sample in data)
            {
                // Get the independent variables (features) and the dependent variable (target)
                float x1 = sample[0];
                float x2 = sample[1];
                float x3 = sample[2];
                float y = sample[3];

                // Prediction and error calculation
                float prediction = a1 * x1 + a2 * x2 + a3 * x3 + b;
                float error = y - prediction;

                // Accumulate squared error for MSE calculation
                sumSquaredError += error * error;

                // Accumulate parts needed for gradient calculation
                // For each parameter 'a', the gradient part is (error * x)
                // For the bias 'b', the gradient part is just the error
                sumErrorForA1 += error * x1;
                sumErrorForA2 += error * x2;
                sumErrorForA3 += error * x3;
                sumErrorForB += error;
            }

            // Number of samples
            int n = data.Length;

            // MSE (Mean Squared Error)
            float meanSquaredError = sumSquaredError / n;

            // Calculate gradients (partial derivatives of MSE, i.e., the "deltas")
            // ∂MSE/∂a1 = -2/n * Σ(error * x1)
            float deltaA1 = -2.0f / n * sumErrorForA1;
            float deltaA2 = -2.0f / n * sumErrorForA2;
            float deltaA3 = -2.0f / n * sumErrorForA3;
            float deltaB = -2.0f / n * sumErrorForB;

            // Update regression parameters using gradient descent
            a1 -= lr * deltaA1;
            a2 -= lr * deltaA2;
            a3 -= lr * deltaA3;
            b -= lr * deltaB;

            if (iteration % printEvery == 0)
            {
                Console.WriteLine($"Iteration: {iteration,6}, MSE: {meanSquaredError,8:F5}, a1: {a1,7:F4}, a2: {a2,7:F4}, a3: {a3,7:F4}, b: {b,7:F4}");
            }
        }

        // 5. Output learned parameters
        Console.WriteLine("\n--- Training Complete ---");
        Console.WriteLine("Learned parameters:");
        Console.WriteLine($"  a1: {a1:F4} (coefficient for 1st variable)");
        Console.WriteLine($"  a2: {a2:F4} (coefficient for 2nd variable)");
        Console.WriteLine($"  a3: {a3:F4} (coefficient for 3rd variable)");
        Console.WriteLine($"  b:  {b:F4} (intercept)");

        Console.WriteLine($"\nExpected parameters from the formula y = 2*x1 + 3*x2 - 1*x3 + 5:");
        Console.WriteLine($"  a1 =  2.0000, a2 =  3.0000, a3 = -1.0000, b = 5.0000");
        Console.ReadLine();
    }
}
