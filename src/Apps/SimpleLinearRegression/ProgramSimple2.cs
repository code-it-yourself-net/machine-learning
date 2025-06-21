// Machine Learning Utils
// File name: ProgramSimple.cs
// Code It Yourself with .NET, 2024

internal class ProgramSimple2
{
    public static void Main()
    {
        // 1. Prepare training data

        // Ground truth coefficients
        float trueA = -2f;
        float trueB = 120f;

        // Number of training samples
        // const int n = 100;

        /*
        // Training data (x) => y = a*x + b
        float[][] data = new float[n][];
        Random rand = new();
        for (int i = 0; i < data.Length; i++)
        {
            // Generate random x in the range [-10, 10]
            float x = (rand.NextSingle() - 0.5f) * 20;
            float y = true_a * x + true_b;
            data[i] = [x, y];
        }
        */
        float[][] data = [
            [10, 100],
            [20, 80],
            [30, 60],
            [40, 40],
            [50, 20],
        ];
        int n = data.Length; // Number of samples

        // 2. Initialize model (weights)
        float a = 0, b = 0;
        const float lr = 0.0005f; // 0.0005f;

        // 3. Training loop

        // Number iterations (epochs)
        const int interations = 30000; //  12_000;

        for (int iteration = 0; iteration < interations; iteration++)
        {
            // Initialize gradients
            float sumErrorValue = 0, sumError = 0;

            // MSE (loss) and squared error
            float meanSquaredError, squaredError = 0;

            foreach (float[] sample in data)
            {
                float x = sample[0];
                float y = sample[1];

                // yHat = prediction
                float yHat = a * x + b;
                float error = y - yHat;

                squaredError += error * error;

                // loss += error * error;

                // Gradients (deltas) according to the provided formulas
                sumErrorValue += error * x;
                sumError += error;
            }

            meanSquaredError = squaredError / n; // Mean Squared Error

            // Apply the -2/n factor to gradients
            float deltaA = -2.0f / n * sumErrorValue;
            float deltaB = -2.0f / n * sumError;

            // Update weights (gradient descent)
            a -= lr * deltaA;
            b -= lr * deltaB;

            //if (iteration % 1000 == 0)
                Console.WriteLine($"Iteration (epoch) {iteration}, mse (loss): {meanSquaredError:F5}, a: {a:F4}, b: {b:F4}");
        }

        // 4. Output learned parameters

        Console.WriteLine($"\nLearned parameters: a = {a:F3}, b = {b:F3}");
        Console.WriteLine($"Expected parameters: a = {trueA}, b = {trueB}");
        Console.ReadLine();
    }
}
