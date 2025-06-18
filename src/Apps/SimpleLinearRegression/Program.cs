// Machine Learning Utils
// File name: Program.cs
// Code It Yourself with .NET, 2024

internal class Program
{
    public static void Main()
    {
        // True parameters for data generation
        double[] trueW = { 3.0, -2.0 }; // A and B
        double trueC = 5.0;             // Bias

        int numSamples = 100;
        double[][] inputs = new double[numSamples][];
        double[] targets = new double[numSamples];
        Random rand = new();

        // Generate training data
        for (int i = 0; i < numSamples; i++)
        {
            double x = rand.NextDouble() * 10;
            double y = rand.NextDouble() * 10;
            double z = trueW[0] * x + trueW[1] * y + trueC;

            inputs[i] = [x, y];
            targets[i] = z;
        }

        // Initialize weight matrix W [2x1] and bias
        double[,] W = new double[2, 1] { { 0.0 }, { 0.0 } }; // weights A and B
        double C = 0.0; // bias

        double learningRate = 0.0005;
        int epochs = 30_000;

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double[,] gradW = new double[2, 1];
            double gradC = 0;
            double loss = 0;

            for (int i = 0; i < numSamples; i++)
            {
                double[] xVec = inputs[i];   // xVec = [x, y]
                double z = targets[i];       // true output

                // Prediction: dot(W.T, xVec) + C
                double pred = W[0, 0] * xVec[0] + W[1, 0] * xVec[1] + C;
                double error = pred - z;

                loss += error * error;

                // Accumulate gradients
                gradW[0, 0] += error * xVec[0];
                gradW[1, 0] += error * xVec[1];
                gradC += error;
            }

            // Average gradients
            gradW[0, 0] /= numSamples;
            gradW[1, 0] /= numSamples;
            gradC /= numSamples;
            loss /= numSamples;

            // Gradient descent update
            W[0, 0] -= learningRate * gradW[0, 0];
            W[1, 0] -= learningRate * gradW[1, 0];
            C -= learningRate * gradC;

            if (epoch % 1000 == 0)
                Console.WriteLine($"Epoch {epoch}, Loss: {loss:F4}, W: [{W[0, 0]:F3}, {W[1, 0]:F3}], C: {C:F3}");
        }

        Console.WriteLine($"\nLearned weights: W = [{W[0, 0]:F3}, {W[1, 0]:F3}], C = {C:F3}");
        Console.WriteLine($"Expected values:  W = [{trueW[0]}, {trueW[1]}], C = {trueC}");
        Console.ReadLine();
    }
}