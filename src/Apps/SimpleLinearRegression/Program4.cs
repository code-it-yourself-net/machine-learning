// Machine Learning Utils
// File name: ProgramSimple.cs
// Code It Yourself with .NET, 2024

internal class Program4
{
    public static void Main()
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        // 1. Set the parameters for the model

        const float lr = 0.0005f;
        const int iterations = 35_000; // 4
        const int printEvery = 1_000; // 1

        // 2. Prepare training data

        float[][] data = [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8]
        ];

        // 3. Initialize model

        float a = 0, b = 0;

        // 4. Training loop

        for (int iteration = 0; iteration < iterations; iteration++)
        {
            // Initialize accumulators for errors
            float sumErrorValue = 0, sumError = 0, squaredError = 0;

            foreach (float[] sample in data)
            {
                float x = sample[0];
                float y = sample[1];

                // Prediction and error calculation
                float prediction = a * x + b;
                float error = y - prediction;

                // Accumulate squared error and gradients
                squaredError += error * error;
                sumErrorValue += error * x;
                sumError += error;
            }

            // Number of samples
            int n = data.Length;

            // MSE
            float meanSquaredError = squaredError / n;

            // Calculate gradients (partial derivatives)
            float deltaA = -2.0f / n * sumErrorValue;
            float deltaB = -2.0f / n * sumError;

            // Update regression parameters
            a -= lr * deltaA;
            b -= lr * deltaB;

            if (iteration % printEvery == 0)
                Console.WriteLine($"Iteration: {iteration,5}, MSE: {meanSquaredError,10:F5}, ∂MSE/∂a: {deltaA,10:F4}, ∂MSE/∂b: {deltaB,10:F4}, a: {a,9:F4}, b: {b,9:F4}");
        }

        // 4. Output learned parameters

        Console.WriteLine($"\nLearned parameters: a = {a:F4}, b = {b:F4}");
        Console.WriteLine($"Expected parameters: a = -2, b = 120");
        Console.ReadLine();
    }
}
