// Machine Learning Utils
// File name: ProgramSimple.cs
// Code It Yourself with .NET, 2024

internal class ProgramSimple
{
    public static void Main()
    {
        // 1. Prepare training data

        // Ground truth coefficients
        float true_a1 = 3.0f;
        float true_a2 = -2.0f;
        float true_b = 5.0f;

        // Number of training samples
        const int n = 100; 

        // Training data (x1, x2) => y = a1*x1 + a2*x2 + b
        float[][] data = new float[n][];
        Random rand = new();
        for (int i = 0; i < data.Length; i++)
        {
            float x1 = rand.NextSingle() * 10;
            float x2 = rand.NextSingle() * 10;
            float y = true_a1 * x1 + true_a2 * x2 + true_b;
            data[i] = [x1, x2, y];
        }

        // 2. Initialize model (weights)
        float a1 = 0, a2 = 0, b = 0;
        float learningRate = 0.0005f;

        // 3. Training loop
        for (int epoch = 0; epoch < 30_000; epoch++)
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
            a1 -= learningRate * delta_a1;
            a2 -= learningRate * delta_a2;
            b -= learningRate * delta_b;

            if (epoch % 1000 == 0)
                Console.WriteLine($"Epoch {epoch}, Loss: {loss:F4}, A1: {a1:F3}, A2: {a2:F3}, B: {b:F3}");
        }

        // 4. Output learned parameters

        Console.WriteLine($"\nLearned parameters: A1 = {a1:F3}, A2 = {a2:F3}, B = {b:F3}");
        Console.WriteLine($"Expected parameters: A1 = {true_a1}, A2 = {true_a2}, B = {true_b}");
        Console.ReadLine();
    }
}
