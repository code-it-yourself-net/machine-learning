// Machine Learning Utils
// File name: ProgramSimple.cs
// Code It Yourself with .NET, 2024

internal class ProgramSimple
{
    public static void Main()
    {
        // 1. Prepare training data

        // Ground truth coefficients
        float true_a1 = 3.1f;
        float true_a2 = -2.4f;
        float true_b = 5.8f;

        // Number of training samples
        const int n = 100; 

        // Training data (x1, x2) => y = a1*x1 + a2*x2 + b
        float[][] data = new float[n][];
        Random rand = new();
        for (int i = 0; i < data.Length; i++)
        {
            // Generate random x1 and x2 in the range [-10, 10]
            float x1 = (rand.NextSingle() - 0.5f) * 20;
            float x2 = (rand.NextSingle() - 0.5f) * 20;
            float y = true_a1 * x1 + true_a2 * x2 + true_b;
            data[i] = [x1, x2, y];
        }

        // 2. Initialize model (weights)
        float a1 = 0, a2 = 0, b = 0;
        const float lr = 0.0005f;

        // 3. Training loop

        // Number iterations (epochs)
        const int interations = 11_000;

        for (int iteration = 0; iteration < interations; iteration++)
        {
            // Initialize gradients
            float delta_a1 = 0, delta_a2 = 0, delta_b = 0;
            float loss = 0;

            foreach (float[] sample in data)
            {
                float x1 = sample[0];
                float x2 = sample[1];
                float y = sample[2];

                // yHat = prediction
                float yHat = a1 * x1 + a2 * x2 + b;
                float error = y - yHat;

                loss += error * error;

                // Gradients (deltas) according to the provided formulas
                delta_a1 += error * x1;
                delta_a2 += error * x2;
                delta_b += error;
            }

            // Apply the -2/n factor to gradients
            delta_a1 = -2.0f / n * delta_a1;
            delta_a2 = -2.0f / n * delta_a2;
            delta_b = -2.0f / n * delta_b;
            loss /= n;

            // Update weights (gradient descent)
            a1 -= lr * delta_a1;
            a2 -= lr * delta_a2;
            b -= lr * delta_b;

            if (iteration % 1000 == 0)
                Console.WriteLine($"Iteration (epoch) {iteration}, loss: {loss:F4}, a1: {a1:F3}, a2: {a2:F3}, b: {b:F3}");
        }

        // 4. Output learned parameters

        Console.WriteLine($"\nLearned parameters: a1 = {a1:F3}, a2 = {a2:F3}, b = {b:F3}");
        Console.WriteLine($"Expected parameters: a1 = {true_a1}, a2 = {true_a2}, b = {true_b}");
        Console.ReadLine();
    }
}
