// Machine Learning Utils
// File name: ProgramSimple.cs
// Code It Yourself with .NET, 2024

internal class ProgramSimple
{
    public static void Main()
    {
        // Ground truth coefficients
        double trueA1 = 3.0;
        double trueA2 = -2.0;
        double trueB = 5.0;

        // Training data (x1, x2) => y = A1*x1 + A2*x2 + B
        double[][] data = new double[100][];
        Random rand = new();
        for (int i = 0; i < data.Length; i++)
        {
            double x1 = rand.NextDouble() * 10;
            double x2 = rand.NextDouble() * 10;
            double y = trueA1 * x1 + trueA2 * x2 + trueB;
            data[i] = [x1, x2, y];
        }

        // Initialize weights
        double A1 = 0, A2 = 0, B = 0;
        double learningRate = 0.0005;

        int n = data.Length;

        // Training loop
        for (int epoch = 0; epoch < 30_000; epoch++)
        {
            double dA1 = 0, dA2 = 0, dB = 0;
            double loss = 0;

            foreach (double[] sample in data)
            {
                double x1 = sample[0];
                double x2 = sample[1];
                double y = sample[2];

                double yHat = A1 * x1 + A2 * x2 + B; // yHat = prediction
                double error = y - yHat; // zgodnie z opisem: (y_i - ŷ_i)

                loss += error * error;

                // Gradients according to the provided formulas
                dA1 += error * x1;
                dA2 += error * x2;
                dB += error;
            }

            // Apply the -2/n factor to gradients
            dA1 = -2.0 / n * dA1;
            dA2 = -2.0 / n * dA2;
            dB = -2.0 / n * dB;
            loss /= n;

            // Update weights (gradient descent)
            A1 -= learningRate * dA1;
            A2 -= learningRate * dA2;
            B -= learningRate * dB;

            if (epoch % 1000 == 0)
                Console.WriteLine($"Epoch {epoch}, Loss: {loss:F4}, A1: {A1:F3}, A2: {A2:F3}, B: {B:F3}");
        }

        Console.WriteLine($"\nLearned parameters: A1 = {A1:F3}, A2 = {A2:F3}, B = {B:F3}");
        Console.WriteLine($"Expected parameters: A1 = {trueA1}, A2 = {trueA2}, B = {trueB}");
        Console.ReadLine();
    }
}
