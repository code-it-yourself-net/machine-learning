// Machine Learning
// File name: Program.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.LearningRates;
using MachineLearning.Typed;
using MachineLearning.Typed.NeuralNetwork.Losses;
using MachineLearning.Typed.NeuralNetwork.Operations;
using MachineLearning.Typed.NeuralNetwork.Optimizers;

using static MachineLearning.Typed.ArrayUtils;

// Prepare data:
// 3 (none, vertical, and horizontal stripes) * 10 examples * 1 channel * 3 rows * 3 columns

const int inputChannels = 1;
const int examples = 30; // 60;
const int outputCategories = 3; // 0, 1, 2

float[,,,] xTrain = new float[examples, inputChannels, 3, 3]
{
    // neither vertical nor horizonta
    {
        {
            { 0, 0, 0 },
            { 0, 0, 0 },
            { 0, 0, 0 }
        }
    },
    {
        {
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 1, 1, 1 }
        }
    },
    {
        {
            { 1, 0, 1 },
            { 0, 1, 0 },
            { 1, 0, 1 }
        }
    },
    {
        {
            { 0, 1, 0 },
            { 1, 1, 1 },
            { 0, 1, 0 },
        }
    },
    {
        {
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 1 },
        }
    },
    // horizontal
    {
        {
            { 1, 1, 1 },
            { 0, 0, 0 },
            { 1, 1, 1 }
        }
    },
    {
        {
            { 1, 1, 1 },
            { 0, 0, 0 },
            { 0, 0, 0 }
        }
    },
    {
        {
            { 0, 0, 0 },
            { 0, 0, 0 },
            { 1, 1, 1 }
        }
    },
    {
        {
            { 0, 0, 0 },
            { 1, 1, 1 },
            { 0, 0, 0 },
        }
    },
    {
        {
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 0, 0, 0 },
        }
    },
    // vertical
    {
        {
            { 1, 0, 1 },
            { 1, 0, 1 },
            { 1, 0, 1 }
        }
    },
    {
        {
            { 0, 1, 0 },
            { 0, 1, 0 },
            { 0, 1, 0 }
        }
    },
    {
        {
            { 1, 0, 0 },
            { 1, 0, 0 },
            { 1, 0, 0 }
        }
    },
    {
        {
            { 0, 0, 1 },
            { 0, 0, 1 },
            { 0, 0, 1 },
        }
    },
    {
        {
            { 0, 1, 1 },
            { 0, 1, 1 },
            { 0, 1, 1 },
        }
    },

    //// negative 

    //// neither vertical nor horizonta
    //{
    //    {
    //        { 0, 0, 0 },
    //        { 0, -1, 0 },
    //        { 0, 0, 0 }
    //    }
    //},
    //{
    //    {
    //        { -1, -1, -1 },
    //        { -1, -1, -1 },
    //        { -1, -1, -1 }
    //    }
    //},
    //{
    //    {
    //        { -1, 0, -1 },
    //        { 0, -1, 0 },
    //        { -1, 0, -1 }
    //    }
    //},
    //{
    //    {
    //        { 0, -1, 0 },
    //        { -1, -1, -1 },
    //        { 0, -1, 0 },
    //    }
    //},
    //{
    //    {
    //        { -1, 0, 2 },
    //        { 0, -1, 0 },
    //        { 2, 0, -1 },
    //    }
    //},
    //// horizontal
    //{
    //    {
    //        { -1, -1, -1 },
    //        { 0, 0, 0 },
    //        { -1, -1, -1 }
    //    }
    //},
    //{
    //    {
    //        { -1, -1, -1 },
    //        { 0, 0, 0 },
    //        { 0, 0, 0 }
    //    }
    //},
    //{
    //    {
    //        { 0, 0, 0 },
    //        { 0, 0, 0 },
    //        { -1, -1, -1 }
    //    }
    //},
    //{
    //    {
    //        { 0, 0, 0 },
    //        { -1, -1, -1 },
    //        { 0, 0, 0 },
    //    }
    //},
    //{
    //    {
    //        { -1, -1, -1 },
    //        { -1, -1, -1 },
    //        { 0, 0, 0 },
    //    }
    //},
    //// vertical
    //{
    //    {
    //        { -1, 0, -1 },
    //        { -1, 0, -1 },
    //        { -1, 0, -1 }
    //    }
    //},
    //{
    //    {
    //        { 0, -1, 0 },
    //        { 0, -1, 0 },
    //        { 0, -1, 0 }
    //    }
    //},
    //{
    //    {
    //        { -1, 0, 0 },
    //        { -1, 0, 0 },
    //        { -1, 0, 0 }
    //    }
    //},
    //{
    //    {
    //        { 0, 0, -1 },
    //        { 0, 0, -1 },
    //        { 0, 0, -1 },
    //    }
    //},
    //{
    //    {
    //        { 0, -1, -1 },
    //        { 0, -1, -1 },
    //        { 0, -1, -1 },
    //    }
    //},

    // 2

    // neither vertical nor horizonta
    {
        {
            { 0, 0, 0 },
            { 0, 0, 0 },
            { 0, 0, 0 }
        }
    },
    {
        {
            { 2, 2, 2 },
            { 2, 2, 2 },
            { 2, 2, 2 }
        }
    },
    {
        {
            { 2, 0, 2 },
            { 0, 2, 0 },
            { 2, 0, 2 }
        }
    },
    {
        {
            { 0, 2, 0 },
            { 2, 2, 2 },
            { 0, 2, 0 },
        }
    },
    {
        {
            { 2, 0, 0 },
            { 0, 2, 0 },
            { 0, 0, 2 },
        }
    },
    // horizontal
    {
        {
            { 2, 2, 2 },
            { 0, 0, 0 },
            { 2, 2, 2 }
        }
    },
    {
        {
            { 2, 2, 2 },
            { 0, 0, 0 },
            { 0, 0, 0 }
        }
    },
    {
        {
            { 0, 0, 0 },
            { 0, 0, 0 },
            { 2, 2, 2 }
        }
    },
    {
        {
            { 0, 0, 0 },
            { 2, 2, 2 },
            { 0, 0, 0 },
        }
    },
    {
        {
            { 2, 2, 2 },
            { 2, 2, 2 },
            { 0, 0, 0 },
        }
    },
    // vertical
    {
        {
            { 2, 0, 2 },
            { 2, 0, 2 },
            { 2, 0, 2 }
        }
    },
    {
        {
            { 0, 2, 0 },
            { 0, 2, 0 },
            { 0, 2, 0 }
        }
    },
    {
        {
            { 2, 0, 0 },
            { 2, 0, 0 },
            { 2, 0, 0 }
        }
    },
    {
        {
            { 0, 0, 2 },
            { 0, 0, 2 },
            { 0, 0, 2 },
        }
    },
    {
        {
            { 0, 2, 2 },
            { 0, 2, 2 },
            { 0, 2, 2 },
        }
    },

    //// negative 

    //// neither vertical nor horizonta
    //{
    //    {
    //        { 0, 0, 0 },
    //        { 0, -2, 0 },
    //        { 0, 0, 0 }
    //    }
    //},
    //{
    //    {
    //        { -2, -2, -2 },
    //        { -2, -2, -2 },
    //        { -2, -2, -2 }
    //    }
    //},
    //{
    //    {
    //        { -2, 0, -2 },
    //        { 0, -2, 0 },
    //        { -2, 0, -2 }
    //    }
    //},
    //{
    //    {
    //        { 0, -2, 0 },
    //        { -2, -2, -2 },
    //        { 0, -2, 0 },
    //    }
    //},
    //{
    //    {
    //        { -2, 0, 0 },
    //        { 0, -2, 0 },
    //        { 0, 0, -2 },
    //    }
    //},
    //// horizontal
    //{
    //    {
    //        { -2, -2, -2 },
    //        { 0, 0, 0 },
    //        { -2, -2, -2 }
    //    }
    //},
    //{
    //    {
    //        { -2, -2, -2 },
    //        { 0, 0, 0 },
    //        { 0, 0, 0 }
    //    }
    //},
    //{
    //    {
    //        { 0, 0, 0 },
    //        { 0, 0, 0 },
    //        { -2, -2, -2 }
    //    }
    //},
    //{
    //    {
    //        { 0, 0, 0 },
    //        { -2, -2, -2 },
    //        { 0, 0, 0 },
    //    }
    //},
    //{
    //    {
    //        { -2, -2, -2 },
    //        { -2, -2, -2 },
    //        { 0, 0, 0 },
    //    }
    //},
    //// vertical
    //{
    //    {
    //        { -2, 0, -2 },
    //        { -2, 0, -2 },
    //        { -2, 0, -2 }
    //    }
    //},
    //{
    //    {
    //        { 0, -2, 0 },
    //        { 0, -2, 0 },
    //        { 0, -2, 0 }
    //    }
    //},
    //{
    //    {
    //        { -2, 0, 0 },
    //        { -2, 0, 0 },
    //        { -2, 0, 0 }
    //    }
    //},
    //{
    //    {
    //        { 0, 0, -2 },
    //        { 0, 0, -2 },
    //        { 0, 0, -2 },
    //    }
    //},
    //{
    //    {
    //        { 0, -2, -2 },
    //        { 0, -2, -2 },
    //        { 0, -2, -2 },
    //    }
    //},
};

// Values - 0 for unknown, 1 for horizontal stripes, 2 for vertical stripes
float[,] oneHot = new float[examples, outputCategories] {
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    //// negative
    //{ 1, 0, 0 },
    //{ 1, 0, 0 },
    //{ 1, 0, 0 },
    //{ 1, 0, 0 },
    //{ 1, 0, 0 },
    //{ 0, 1, 0 },
    //{ 0, 1, 0 },
    //{ 0, 1, 0 },
    //{ 0, 1, 0 },
    //{ 0, 1, 0 },
    //{ 0, 0, 1 },
    //{ 0, 0, 1 },
    //{ 0, 0, 1 },
    //{ 0, 0, 1 },
    //{ 0, 0, 1 },
    // 2
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    //// negative
    //{ 1, 0, 0 },
    //{ 1, 0, 0 },
    //{ 1, 0, 0 },
    //{ 1, 0, 0 },
    //{ 1, 0, 0 },
    //{ 0, 1, 0 },
    //{ 0, 1, 0 },
    //{ 0, 1, 0 },
    //{ 0, 1, 0 },
    //{ 0, 1, 0 },
    //{ 0, 0, 1 },
    //{ 0, 0, 1 },
    //{ 0, 0, 1 },
    //{ 0, 0, 1 },
    //{ 0, 0, 1 },
};

// Scale xTrain and xTest to mean 0, variance 1
Console.WriteLine("Scale data to mean 0...");

float mean = xTrain.Mean();
xTrain.AddInPlace(-mean);

Console.WriteLine($"xTrain min: {xTrain.Min()}");
Console.WriteLine($"xTrain max: {xTrain.Max()}");

Console.WriteLine("\nScale data to variance 1...");

float std = xTrain.Std();
xTrain.DivideInPlace(std);

Console.WriteLine($"xTrain min: {xTrain.Min()}");
Console.WriteLine($"xTrain max: {xTrain.Max()}");

float loss = Train(xTrain, oneHot, 3_000, 2411240);

Console.WriteLine($"loss: {loss}");
Console.ReadLine();

static float Train(float[,,,] xTrain, float[,] yTrain, int iterations /*, float learningRate = 0.01f */, int? seed = null)
{
    float loss = 0;
    Random random;
    if (seed.HasValue)
        random = new(seed.Value);
    else
        random = new();

    const int outputChannels = 4;
    float[,,,] kernels = CreateRandom(inputChannels, outputChannels, 3, 3, random);
    Conv2D conv2D = new(kernels);
    Tanh4D tanh4D = new();
    Flatten flatten = new();

    float[,] weights = CreateRandom(outputChannels * 3 * 3, outputCategories, random);
    WeightMultiply weightMultiply = new(weights);

    float[] bias = CreateRandom(outputCategories, random);
    BiasAdd biasAdd = new(bias);

    SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss = new();

    LearningRate learningRate = new ExponentialDecayLearningRate(0.02f, 0.001f);
    Optimizer optimizer = new StochasticGradientDescentMomentum(learningRate, 0.9f);

    for (int i = 0; i < iterations; i++)
    {
        (xTrain, yTrain) = PermuteData(xTrain, yTrain, random);

        // Forward

        float[,,,] conv2DOutput = conv2D.Forward(xTrain, false);
        float[,,,] tanh4DOutput = tanh4D.Forward(conv2DOutput, false);
        float[,] flattenOutput = flatten.Forward(tanh4DOutput, false);
        float[,] weightMultiplyOutput = weightMultiply.Forward(flattenOutput, false);
        float[,] biasAddOutput = biasAdd.Forward(weightMultiplyOutput, false);

        loss = softmaxCrossEntropyLoss.Forward(biasAddOutput, yTrain);

        // Print the loss every 10 steps

        if (i % 10 == 0)
        {
            Console.WriteLine($"iteration: {i}, loss: {loss}");

            // Write the percent of the correct predictions
            float[,] prediction = softmaxCrossEntropyLoss.Prediction;
            int[] predictionArgmax = prediction.Argmax();
            int rows = predictionArgmax.GetLength(0);
            int hits = 0;
            for (int row = 0; row < rows; row++)
            {
                int predictedDigit = predictionArgmax[row];
                if (yTrain[row, predictedDigit] == 1f)
                    hits++;
            }
            float accuracy = (float)hits / rows;
            Console.WriteLine($"accuracy: {accuracy}");

            // TODO: display conv2D.Param to see the changes in the kernels
        }

        // Backward

        float[,] softmaxCrossEntropyLossGradient = softmaxCrossEntropyLoss.Backward();

        float[,] biasAddGradient = biasAdd.Backward(softmaxCrossEntropyLossGradient);
        float[,] weightMultiplyGradient = weightMultiply.Backward(biasAddGradient);
        float[,,,] flattenGradient = flatten.Backward(weightMultiplyGradient);
        float[,,,] tanh4DGradient = tanh4D.Backward(flattenGradient);
        float[,,,] conv2DGradient = conv2D.Backward(tanh4DGradient);

        // Update weights

        optimizer.UpdateLearningRate(i + 1, iterations);
        conv2D.UpdateParams(null, optimizer);
        weightMultiply.UpdateParams(null, optimizer);
        biasAdd.UpdateParams(null, optimizer);
    }

    return loss;
}

