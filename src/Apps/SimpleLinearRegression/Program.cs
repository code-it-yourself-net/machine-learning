// Machine Learning Utils
// File name: Class1.cs
// Code It Yourself with .NET, 2024


// Ground truth coefficients
double trueA = 3.0;
double trueB = -2.0;
double trueC = 5.0;

// Training data (x, y) => z = A*x + B*y + C
double[][] data = new double[100][];
Random rand = new();
for (int i = 0; i < data.Length; i++)
{
    double x = rand.NextDouble() * 10;
    double y = rand.NextDouble() * 10;
    double z = trueA * x + trueB * y + trueC;
    data[i] = new double[] { x, y, z };
}

// Initialize weights
double A = 0, B = 0, C = 0;
double learningRate = 0.0005;

// Training loop
for (int epoch = 0; epoch < 30_000; epoch++)
{
    double dA = 0, dB = 0, dC = 0;
    double loss = 0;

    foreach (double[] sample in data)
    {
        double x = sample[0];
        double y = sample[1];
        double z = sample[2];

        double pred = A * x + B * y + C;
        double error = pred - z;

        loss += error * error;

        // Compute gradients
        dA += error * x;
        dB += error * y;
        dC += error;
    }

    // Average gradients
    dA /= data.Length;
    dB /= data.Length;
    dC /= data.Length;
    loss /= data.Length;

    // Update weights (gradient descent)
    A -= learningRate * dA;
    B -= learningRate * dB;
    C -= learningRate * dC;

    if (epoch % 1000 == 0)
        Console.WriteLine($"Epoch {epoch}, Loss: {loss:F4}, A: {A:F3}, B: {B:F3}, C: {C:F3}");
}

Console.WriteLine($"\nLearned parameters: A = {A:F3}, B = {B:F3}, C = {C:F3}");
Console.WriteLine($"Expected parameters: A = {trueA}, B = {trueB}, C = {trueC}");
Console.ReadLine();