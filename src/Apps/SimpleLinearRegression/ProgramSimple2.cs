// Machine Learning Utils
// File name: ProgramSimple.cs
// Code It Yourself with .NET, 2024

internal class ProgramSimple2
{
    public static void Main()
    {
        // 1. Prepare training data

        // Ground truth coefficients
        float true_a = -2f;
        float true_b = 120f;

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
            float error_times_value_sum = 0, error_sum = 0;

            // MSE (loss) and squared error
            float mse, error_squared_sum = 0;

            foreach (float[] sample in data)
            {
                float x = sample[0];
                float y = sample[1];

                // yHat = prediction
                float yHat = a * x + b;
                float error = y - yHat;

                error_squared_sum += error * error;

                // loss += error * error;

                // Gradients (deltas) according to the provided formulas
                error_times_value_sum += error * x;
                error_sum += error;
            }

            mse = error_squared_sum / n; // Mean Squared Error

            // Apply the -2/n factor to gradients
            float delta_a = -2.0f / n * error_times_value_sum;
            float delta_b = -2.0f / n * error_sum;

            // Update weights (gradient descent)
            a -= lr * delta_a;
            b -= lr * delta_b;

            //if (iteration % 1000 == 0)
                Console.WriteLine($"Iteration (epoch) {iteration}, mse (loss): {mse:F5}, a: {a:F4}, b: {b:F4}");
        }

        // 4. Output learned parameters

        Console.WriteLine($"\nLearned parameters: a = {a:F3}, b = {b:F3}");
        Console.WriteLine($"Expected parameters: a = {true_a}, b = {true_b}");
        Console.ReadLine();
    }
}
