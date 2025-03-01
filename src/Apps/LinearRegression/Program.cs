// Machine Learning
// File name: Program.cs
// Code It Yourself with .NET, 2024

// Based on the code from Deep Learning from Scratch: Building with Python from First Principles by Seth Weidman
// https://github.com/SethHWeidman/DLFS_code/blob/master/02_fundamentals/Code.ipynb

using System.Diagnostics;

using MachineLearning;

// main method

Console.WriteLine("Linear function");
Matrix xTrain = new(new float[,] { { 10, 20 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 }, { 0, 0 }, { -5, -6 }, { -100f, 2f } });
Matrix yTrain = new(new float[,] { { -30 }, { -5 }, { -7 }, { -9 }, { -11 }, { 0 }, { 11 }, { 98f } });

(Matrix weights, float bias, float loss) = Train(xTrain, yTrain, iterations: 1_500, learningRate: 0.00001f, batchSize: 10);

Console.WriteLine();
Console.WriteLine($"weights: \n{weights}");
Console.WriteLine($"bias: {bias}");
Console.WriteLine($"loss: {loss}");
Console.WriteLine();

for (int row = 0; row < xTrain.GetDimension(Dimension.Rows); row++)
{
    Matrix x = xTrain.GetRow(row);
    Matrix y = x.MultiplyDot(weights).Add(bias);
    Console.WriteLine($"x: {x.Array.GetValue(0, 0)}, {x.Array.GetValue(0, 1)} y: {y.Array.GetValue(0, 0)}");
}

Console.ReadLine();

// Functions

static (Matrix n, Matrix p, float loss) ForwardLinearRegression(Matrix xBatch, Matrix yBatch, Matrix weights, float bias)
{
    Debug.Assert(xBatch.GetDimension(Dimension.Rows) == yBatch.GetDimension(Dimension.Rows));

    Debug.Assert(xBatch.GetDimension(Dimension.Columns) == weights.GetDimension(Dimension.Rows));

    // Calculate the values of the input elements multiplied by the weights.
    Matrix n = xBatch.MultiplyDot(weights);

    // Add the bias to the values to make the predictions.
    Matrix p = n.Add(bias);

    // Calculate the mean squared error loss.
    float loss = yBatch.Subtract(p).Power(2).Mean();

    return (n, p, loss);
}

// bias is not used here, because it's a scalar (so we don't need to know any dimensions) and its derivative is just equal to 1
static (Matrix weightsLossGradient, float biasLossGradient) LossGradients(Matrix xBatch, Matrix yBatch, Matrix weights, float bias, Matrix n, Matrix p)
{
    int batchSize = xBatch.GetDimension(Dimension.Rows);

    // Calculate the derivate of loss with respect to predictions.
    Matrix dLdP = yBatch.Subtract(p).Multiply(-2f / batchSize);

    // Calculate the derivate of predictions with respect to n.
    Matrix dPdN = Matrix.Ones(n);

    // Calculate the derivate of bias with respect to n.
    float dPdBias = 1;

    // Calculate the derivate of loss with respect to n.
    Matrix dLdN = dLdP.MultiplyElementwise(dPdN);

    // Calculate the derivate of n with respect to weights.
    Matrix dNdW = xBatch.Transpose();

    Matrix dLdW = dNdW.MultiplyDot(dLdN);

    Matrix dLdPxdPdBias = dLdP.Multiply(dPdBias);

    // Calculate the derivate of loss with respect to bias.
    float dLdBias = dLdPxdPdBias.Sum();

    return (dLdW, dLdBias);
}

static (Matrix xPermuted, Matrix yPermuted) PermuteData(Matrix x, Matrix y, Random random)
{
    Debug.Assert(x.GetDimension(Dimension.Rows) == y.GetDimension(Dimension.Rows));

    int[] indices = [.. Enumerable.Range(0, x.GetDimension(Dimension.Rows)).OrderBy(i => random.Next())];

    Matrix xPermuted = Matrix.Zeros(x);
    Matrix yPermuted = Matrix.Zeros(y);

    for (int i = 0; i < x.GetDimension(Dimension.Rows); i++)
    {
        //xPermuted[i] = x[indices[i]];
        //yPermuted[i] = y[indices[i]];
        xPermuted.SetRow(i, x.GetRow(indices[i]));
        yPermuted.SetRow(i, y.GetRow(indices[i]));
    }

    return (xPermuted, yPermuted);
}

static (Matrix weights, float bias, float loss) Train(Matrix xTrain, Matrix yTrain, int iterations = 200, float learningRate = 0.01f, int? seed = null, int batchSize = 100)
{
    float loss = 0;
    Random random;
    if (seed.HasValue)
        random = new(seed.Value);
    else
        random = new();

    Matrix weights = Matrix.Random(xTrain.GetDimension(Dimension.Columns), 1, random);
    float bias = random.NextSingle() - 0.5f;

    // Uncomment for tests:
    // Matrix weights = new(new float[2, 1] { { -1f }, { 1f } });
    // float bias = 0;

    int batchStart = int.MaxValue;

    int xTrainRows = xTrain.GetDimension(Dimension.Rows);

    for (int i = 0; i < iterations; i++)
    {
        if (batchStart >= xTrainRows)
        {
            (xTrain, yTrain) = PermuteData(xTrain, yTrain, random);
            batchStart = 0;
        }

        int effectiveBatchSize = Math.Min(batchSize, xTrainRows - batchStart);
        int batchEnd = effectiveBatchSize + batchStart;
        Matrix xBatch = xTrain.GetRows(batchStart..batchEnd);
        Matrix yBatch = yTrain.GetRows(batchStart..batchEnd);

        batchStart += effectiveBatchSize;

        (Matrix n, Matrix p, loss) = ForwardLinearRegression(xBatch, yBatch, weights, bias);

        // Print loss every 100 steps
        if (i % 100 == 0)
        {
            Console.WriteLine($"iteration: {i}, loss: {loss}");
        }

        (Matrix weightsLossGradient, float biasLossGradient) = LossGradients(xBatch, yBatch, weights, bias, n, p);

        weights = weights.Subtract(weightsLossGradient.Multiply(learningRate));

        bias -= biasLossGradient * learningRate;
    }

    return (weights, bias, loss);
}

