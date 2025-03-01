// Machine Learning Utils
// File name: Program.cs
// Code It Yourself with .NET, 2024

// Based on the code from Deep Learning from Scratch: Building with Python from First Principles by Seth Weidman
// https://github.com/SethHWeidman/DLFS_code/blob/master/02_fundamentals/Code.ipynb

using System.Diagnostics;

using MachineLearning;

// main method

Console.WriteLine("Linear function");
Matrix xTrain = new(new float[,] { { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 }, { 0, 0 }, { -2, -6 }, { -3, -2 }, { 1, 6 }, { -3, 0 }, { 3, 0 }, { -2, 0 }, { -2.5f, 1 }, { -4, 5 }, { -5, 7 }, { -1, 3 }, { -3, 1 } });
// y = x1 + x2 + 0
// Matrix yTrain = new(new float[,] { { -3 }, { -5 }, { -7 }, { -9 }, { -11 }, { 0 }, { 11 }, { 9.8f } });
// y = 2*x1^2 - 6*x2 + 4.5
Matrix yTrain = new(new float[,] { { -5.5f }, { -5.5f }, { -1.5f }, { 6.5f }, { 18.5f }, { 4.5f }, { 48.5f }, { -10.5f }, { -29.5f }, { 22.5f }, { 22.5f }, { 12.5f }, { 11 }, { 6.5f }, { 12.5f }, { -11.5f }, { 16.5f } });

(Matrix weights1, Matrix weights2, Matrix bias1, float bias2, float loss) = Train(xTrain, yTrain, 4, iterations: 39_500, learningRate: 0.0015f, batchSize: 100);

Console.WriteLine();
Console.WriteLine($"weights1: \n{weights1}");
Console.WriteLine($"weights2: \n{weights2}");
Console.WriteLine($"bias1: \n{bias1}");
Console.WriteLine($"bias2: {bias2}");
Console.WriteLine($"loss: {loss}");
Console.WriteLine();

for (int row = 0; row < xTrain.GetDimension(Dimension.Rows); row++)
{
    Matrix x = xTrain.GetRow(row);
    Matrix y = GetPrediction(x, weights1, weights2, bias1, bias2);
    Console.WriteLine($"x: {x.Array.GetValue(0, 0)}, {x.Array.GetValue(0, 1)}, p: {y.Array.GetValue(0, 0)}, y: {yTrain.Array.GetValue(row, 0)}");
}

Console.WriteLine("Eval");
Matrix eval = new(new float[,] { { -3f, 0f }, { -2.5f, 0f }, { -2f, 0f }, { -1.5f, 0f }, { -1f, 0f }, { -0.5f, 0f }, { 0f, 0f }, { 0.5f, 0f }, { 1f, 0f }, { 1.5f, 0f }, { 2f, 0f }, { 2.5f, 0f }, { 3f, 0f } });

for (int row = 0; row < eval.GetDimension(Dimension.Rows); row++)
{
    Matrix x = eval.GetRow(row);
    Matrix y = GetPrediction(x, weights1, weights2, bias1, bias2);
    Console.WriteLine($"x: {x.Array.GetValue(0, 0)}, {x.Array.GetValue(0, 1)}, p: {y.Array.GetValue(0, 0)}");
}

Console.ReadLine();

// weights1 - weights for the first layer [ inputSize (no of columns/attributes of X) x hiddenSize ]
// weights2 - weights for the second layer [ hiddenSize x 1 ]
// bias1 (matrix) - bias for the first layer (for every neuron in the first layer)
// bias2 (scalar) - bias for the second layer (there is only one neuron in the second layer)
// p - predictions (m2 + bias2)
// m1 - input multiplied by weights1
// n1 - input multiplied by weights1 plus bias1
// o1 - result of the activation function applied to (input multiplied by weights1 plus bias1)
// m2 - result of o1 (result of the activation function from the first layer) multiplied by weights2
static (Matrix m1, Matrix n1, Matrix o1, Matrix m2, Matrix p, float loss) ForwardLoss(Matrix xBatch, Matrix yBatch, Matrix weights1, Matrix weights2, Matrix bias1, float bias2)
{
    Debug.Assert(xBatch.GetDimension(Dimension.Rows) == yBatch.GetDimension(Dimension.Rows));
    Debug.Assert(xBatch.GetDimension(Dimension.Columns) == weights1.GetDimension(Dimension.Rows));

    // The first layer.
    Matrix m1 = xBatch.MultiplyDot(weights1);
    Matrix n1 = m1.AddRow(bias1);
    Matrix o1 = n1.Sigmoid();

    // The second layer.
    Matrix m2 = o1.MultiplyDot(weights2);
    Matrix p = m2.Add(bias2);

    float loss = yBatch.Subtract(p).Power(2).Mean();

    return (m1, n1, o1, m2, p, loss);
}

static Matrix GetPrediction(Matrix x, Matrix weights1, Matrix weights2, Matrix bias1, float bias2)
{
    Matrix m1 = x.MultiplyDot(weights1);
    Matrix n1 = m1.Add(bias1);
    Matrix o1 = n1.Sigmoid();
    Matrix m2 = o1.MultiplyDot(weights2);
    Matrix p = m2.Add(bias2);
    return p;
}

// weights1 - weights for the first layer [ inputSize (no of columns/attributes of X) x hiddenSize ]
// weights2 - weights for the second layer [ hiddenSize x 1 ]
// bias1 (matrix) - bias for the first layer (for every neuron in the first layer)
// bias2 (scalar) - bias for the second layer (there is only one neuron in the second layer); bias2 is not used here, because it's a scalar (so we don't need to know any dimensions) and its derivative is just equal to 1
// p - predictions
// m1 - input multiplied by weights1
// n1 - input multiplied by weights1 plus bias1
// o1 - result of the activation function applied to (input multiplied by weights1 plus bias1)
// m2 - result of o1 (o1 is the result of the activation function from the first layer) multiplied by weights2
static (Matrix weights1LossGradient, Matrix weights2LossGradient, Matrix bias1LossGradient, float bias2LossGradient) LossGradients(Matrix xBatch, Matrix yBatch, Matrix weights1, Matrix weights2, Matrix bias1, float bias2, Matrix m1, Matrix n1, Matrix o1, Matrix m2, Matrix p)
{
    // The first layer.
    int batchSize = xBatch.GetDimension(Dimension.Rows);
    Matrix dLdP = yBatch.Subtract(p).Multiply(-2f / batchSize);
    Matrix dPdM2 = Matrix.Ones(m2);
    Matrix dLdM2 = dLdP.MultiplyElementwise(dPdM2);
    float dPdBias2 = 1;
    float dLBias2 = dLdP.Multiply(dPdBias2).Sum();
    Matrix dM2dW2 = o1.Transpose();
    Matrix dLdW2 = dM2dW2.MultiplyDot(dLdP);

    // The second layer.
    Matrix dM2dO1 = weights2.Transpose();
    Matrix dLdO1 = dLdM2.MultiplyDot(dM2dO1);
    Matrix dO1dN1 = n1.SigmoidDerivative();
    Matrix dLdN1 = dLdO1.MultiplyElementwise(dO1dN1);
    Matrix dN1dBias1 = Matrix.Ones(bias1);
    Matrix dN1dM1 = Matrix.Ones(m1);
    Matrix dLdBias1 = dLdN1.MultiplyElementwise(dN1dBias1).SumBy(Dimension.Rows);
    Matrix dLdM1 = dLdN1.MultiplyElementwise(dN1dM1);
    Matrix dM1dW1 = xBatch.Transpose();
    Matrix dLdW1 = dM1dW1.MultiplyDot(dLdM1);

    return (dLdW1, dLdW2, dLdBias1, dLBias2);
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

static (Matrix weights1, Matrix weights2, Matrix bias1, float bias2, float loss) Train(Matrix xTrain, Matrix yTrain, int hiddenSize, int iterations = 2_000, float learningRate = 0.01f, int? seed = null, int batchSize = 100)
{
    float loss = 0;
    Random random;
    if (seed.HasValue)
        random = new(seed.Value);
    else
        random = new();

    int inputSize = xTrain.GetDimension(Dimension.Columns);
    Matrix weights1 = Matrix.Random(inputSize, hiddenSize, random);
    Matrix bias1 = Matrix.Random(1, hiddenSize, random);

    Matrix weights2 = Matrix.Random(hiddenSize, 1, random);
    float bias2 = random.NextSingle() - 0.5f;

    (xTrain, yTrain) = PermuteData(xTrain, yTrain, random);

    int batchStart = 0;

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

        (Matrix m1, Matrix n1, Matrix o1, Matrix m2, Matrix p, loss) = ForwardLoss(xBatch, yBatch, weights1, weights2, bias1, bias2);

        // Print loss every 100 steps
        if (i % 100 == 0)
        {
            Console.WriteLine($"iteration: {i}, loss: {loss}");
        }

        (Matrix weights1LossGradient, Matrix weights2LossGradient, Matrix bias1LossGradient, float bias2LossGradient) = LossGradients(xBatch, yBatch, weights1, weights2, bias1, bias2, m1, n1, o1, m2, p);

        weights1 = weights1.Subtract(weights1LossGradient.Multiply(learningRate));
        weights2 = weights2.Subtract(weights2LossGradient.Multiply(learningRate));
        bias1 = bias1.Subtract(bias1LossGradient.Multiply(learningRate));
        bias2 -= bias2LossGradient * learningRate;
    }

    return (weights1, weights2, bias1, bias2, loss);
}
