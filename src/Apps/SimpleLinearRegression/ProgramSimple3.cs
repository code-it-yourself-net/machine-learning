// Machine Learning Utils
// File name: ProgramSimple.cs
// Code It Yourself with .NET, 2024

internal class ProgramSimple3
{
    public static void Main()
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        // 1. Set the parameters for the model

        const float lr = 0.0005f;
        const int interations = 4; //  35_000;
        const int printEvery = 1; //  1_000;

        // 2. Prepare training data

        float[][] data = [
            [10, 100],
            [20, 80],
            [30, 60],
            [40, 40],
            [50, 20],
        ];

        // Number of samples
        int n = data.Length; 

        // 3. Initialize model (weights)

        float a = 0, b = 0;

        // 4. Training loop

        for (int iteration = 0; iteration < interations; iteration++)
        {
            // Initialize gradients
            float sumErrorValue = 0, sumError = 0;

            // MSE and squared error
            float meanSquaredError, squaredError = 0;

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

            // MSE
            meanSquaredError = squaredError / n;

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

        Console.WriteLine($"\nLearned parameters: a = {a:F3}, b = {b:F3}");
        Console.WriteLine($"Expected parameters: a = -2, b = 120");
        Console.ReadLine();
    }
}
