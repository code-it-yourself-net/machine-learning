// Machine Learning Utils
// File name: ProgramSimple.cs
// Code It Yourself with .NET, 2024

internal class ProgramSimple
{
    public static void Main()
    {
        // 1. Prepare training data

        // Ground truth coefficients
        float trueA1 = 3.0f;
        float trueA2 = -2.0f;
        float trueB = 5.0f;

        // Number of training samples
        const int n = 100; 

        // Training data (x1, x2) => y = A1*x1 + A2*x2 + B
        float[][] data = new float[n][];
        Random rand = new();
        for (int i = 0; i < data.Length; i++)
        {
            float x1 = rand.NextSingle() * 10;
            float x2 = rand.NextSingle() * 10;
            float y = trueA1 * x1 + trueA2 * x2 + trueB;
            data[i] = [x1, x2, y];
        }

        // 2. Initialize model (weights)
        float a1 = 0, a2 = 0, b = 0;
        float learningRate = 0.0005f;

        // 3. Training loop
        for (int epoch = 0; epoch < 30_000; epoch++)
        {
            float dA1 = 0, dA2 = 0, dB = 0;
            float loss = 0;

            foreach (float[] sample in data)
            {
                float x1 = sample[0];
                float x2 = sample[1];
                float y = sample[2];

                float yHat = a1 * x1 + a2 * x2 + b; // yHat = prediction
                float error = y - yHat; // zgodnie z opisem: (y_i - ŷ_i)

                loss += error * error;

                // Gradients according to the provided formulas
                dA1 += error * x1;
                dA2 += error * x2;
                dB += error;
            }

            // Apply the -2/n factor to gradients
            dA1 = -2.0f / n * dA1;
            dA2 = -2.0f / n * dA2;
            dB = -2.0f / n * dB;
            loss /= n;

            // Update weights (gradient descent)
            a1 -= learningRate * dA1;
            a2 -= learningRate * dA2;
            b -= learningRate * dB;

            if (epoch % 1000 == 0)
                Console.WriteLine($"Epoch {epoch}, Loss: {loss:F4}, A1: {a1:F3}, A2: {a2:F3}, B: {b:F3}");
        }

        // 4. Output learned parameters

        Console.WriteLine($"\nLearned parameters: A1 = {a1:F3}, A2 = {a2:F3}, B = {b:F3}");
        Console.WriteLine($"Expected parameters: A1 = {trueA1}, A2 = {trueA2}, B = {trueB}");
        Console.ReadLine();
    }
}
