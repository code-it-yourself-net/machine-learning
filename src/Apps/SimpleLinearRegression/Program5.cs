// Machine Learning Utils
// File name: Program5.cs
// Code It Yourself with .NET, 2024

internal class Program5
{
    public static void Main()
    {

        // 1. Set the parameters for the model
        const int numFeatures = 3; // Number of independent variables (a1, a2, a3)
        const float lr = 0.001f;   // Learning Rate - adjusted for this problem
        const int iterations = 50_000;
        const int printEvery = 2_000;

        // 2. Prepare training data
        // Each inner array represents a sample: [x1, x2, x3, y]
        // We are trying to find the relationship: y = 2*x1 + 3*x2 - 1*x3 + 5
        float[][] data = [
            [1, 2, 1, 12], // y = 2*1 + 3*2 - 1*1 + 5 = 12
    [2, 1, 2, 10], // y = 2*2 + 3*1 - 1*2 + 5 = 10
    [3, 3, 1, 19], // y = 2*3 + 3*3 - 1*1 + 5 = 19
    [4, 2, 3, 16], // y = 2*4 + 3*2 - 1*3 + 5 = 16
    [1, 4, 2, 17]  // y = 2*1 + 3*4 - 1*2 + 5 = 17
        ];

        // 3. Initialize model parameters
        // We now have weights for each feature and one bias term.
        float[] weights = new float[numFeatures]; // Corresponds to w1, w2, w3
        float bias = 0;                          // Corresponds to b

        // Initialize with small random values or zeros
        // for (int i = 0; i < numFeatures; i++) weights[i] = 0; // Already initialized to 0

        // 4. Training loop
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            // Initialize accumulators for errors and gradients for this iteration
            float sumSquaredError = 0;
            float[] sumErrorForWeights = new float[numFeatures]; // Accumulator for each weight's gradient part
            float sumErrorForBias = 0;                           // Accumulator for the bias's gradient part

            foreach (float[] sample in data)
            {
                // Separate features (x) from the target (y)
                float[] x = new float[numFeatures];
                for (int i = 0; i < numFeatures; i++)
                {
                    x[i] = sample[i];
                }
                float y = sample[numFeatures];

                // Prediction and error calculation
                // prediction = w1*x1 + w2*x2 + w3*x3 + b
                float prediction = bias;
                for (int i = 0; i < numFeatures; i++)
                {
                    prediction += weights[i] * x[i];
                }

                float error = y - prediction;

                // Accumulate squared error for MSE calculation
                sumSquaredError += error * error;

                // Accumulate parts needed for gradient calculation
                // For each weight wi, the gradient part is (error * xi)
                for (int i = 0; i < numFeatures; i++)
                {
                    sumErrorForWeights[i] += error * x[i];
                }
                // For the bias, the gradient part is just the error
                sumErrorForBias += error;
            }

            // Number of samples
            int n = data.Length;

            // MSE (Mean Squared Error)
            float meanSquaredError = sumSquaredError / n;

            // Calculate gradients (partial derivatives of MSE)
            // ∂MSE/∂wi = -2/n * Σ(error * xi)
            float[] deltaWeights = new float[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                deltaWeights[i] = -2.0f / n * sumErrorForWeights[i];
            }

            // ∂MSE/∂b = -2/n * Σ(error)
            float deltaBias = -2.0f / n * sumErrorForBias;

            // Update regression parameters using gradient descent
            for (int i = 0; i < numFeatures; i++)
            {
                weights[i] -= lr * deltaWeights[i];
            }
            bias -= lr * deltaBias;

            if (iteration % printEvery == 0)
            {
                // Format the output string for weights
                string weightsStr = string.Join(", ", weights.Select(w => $"{w,9:F4}"));
                Console.WriteLine($"Iteration: {iteration,6}, MSE: {meanSquaredError,10:F5}, Weights: [{weightsStr}], Bias: {bias,9:F4}");
            }
        }

        // 5. Output learned parameters
        Console.WriteLine("\n--- Training Complete ---");
        Console.WriteLine($"Learned parameters:");
        for (int i = 0; i < numFeatures; i++)
        {
            Console.WriteLine($"  w{i + 1} (for a{i + 1}): {weights[i]:F4}");
        }
        Console.WriteLine($"  b (bias):   {bias:F4}");

        Console.WriteLine($"\nExpected parameters from the formula y = 2*x1 + 3*x2 - 1*x3 + 5:");
        Console.WriteLine($"  w1 =  2.0000, w2 =  3.0000, w3 = -1.0000, b = 5.0000");
        Console.ReadLine();
    }
}
