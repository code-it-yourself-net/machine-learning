// Machine Learning
// File name: FirstNeuralNetwork.cs
// Code It Yourself with .NET, 2024

using MachineLearning;
using MachineLearning.NeuralNetwork;
using MachineLearning.NeuralNetwork.DataSources;
using MachineLearning.NeuralNetwork.Layers;
using MachineLearning.NeuralNetwork.LearningRates;
using MachineLearning.NeuralNetwork.Losses;
using MachineLearning.NeuralNetwork.Operations;
using MachineLearning.NeuralNetwork.Optimizers;
using MachineLearning.NeuralNetwork.ParamInitializers;

namespace NeuralNetworkTests;

internal class FirstNeuralNetwork
{
    private static void Main(string[] args)
    {
        float[,] arguments = new float[,] { { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 }, { 0, 0 }, { -2, -6 }, { -3, -2 }, { 1, 6 }, { -3, 0 }, { 3, 0 }, { -2, 0 }, { -2.5f, 1 }, { -4, 5 }, { -5, 7 }, { -1, 3 }, { -3, 1 }, { -.5f, 8 }, { -1, 4 }, { .5f, 7 }, { 5, .1f }, { 1, -1 }, { 2, -3 } };

        // y = 2 * x1 ^ 2 - 6 * x2 + 4.5
        // FunctionDataSource dataSource = new(arguments, (float[] x) => 2f * x[0] * x[0] - 6f * x[1] + 4.5f, 0);

        // y = -(x1 + x2)
        FunctionDataSource dataSource = new(arguments, (float[] x) => -x[0] - x[1], 0);

        //ParamInitializer rangeInitializer = new RangeInitializer(-2f, 2f);
        ParamInitializer randomInitializer = new RandomInitializer();

        NeuralNetwork neuralNetwork = new(
            layers: [
                new DenseLayer(4, new Sigmoid(), randomInitializer),
                new DenseLayer(1, new Linear(), randomInitializer)
            ],
            lossFunction: new MeanSquaredError()
        );

        LearningRate learningRate = new ConstantLearningRate(0.002f);
        Trainer trainer = new(neuralNetwork, new StochasticGradientDescent(learningRate), ConsoleOutputMode.OnlyOnEval);
        trainer.Fit(dataSource, batchSize: 32, epochs: 10_000, evalEveryEpochs: 1_000);

        Matrix[] @params = neuralNetwork.GetAllParams();
        Matrix weights1 = @params[0];
        Matrix bias1 = @params[1];
        Matrix weights2 = @params[2];
        Matrix bias2 = @params[3];
        float loss = neuralNetwork.LastLoss;

        Console.WriteLine();
        Console.WriteLine($"weights1: \n{weights1}");
        Console.WriteLine($"bias1: {bias1}");
        Console.WriteLine($"weights2: \n{weights2}");
        Console.WriteLine($"bias2: {bias2}");
        Console.WriteLine($"loss: {loss}");
        Console.WriteLine();

        (Matrix xTrain, Matrix yTrain) = dataSource.GetAllData();
        for (int row = 0; row < xTrain.GetDimension(Dimension.Rows); row++)
        {
            Matrix x = xTrain.GetRow(row);
            Matrix y = neuralNetwork.Forward(x, true);
            Console.WriteLine($"x: {x.Array.GetValue(0, 0)}, {x.Array.GetValue(0, 1)}, p: {y.Array.GetValue(0, 0)}, y: {yTrain.Array.GetValue(row, 0)}");
        }

        Console.WriteLine("Eval");
        Matrix eval = new(new float[,] { { -3f, 0f }, { -2.5f, 0f }, { -2f, 0f }, { -1.5f, 0f }, { -1f, 0f }, { -0.5f, 0f }, { 0f, 0f }, { 0.5f, 0f }, { 1f, 0f }, { 1.5f, 0f }, { 2f, 0f }, { 2.5f, 0f }, { 3f, 0f } });

        for (int row = 0; row < eval.GetDimension(Dimension.Rows); row++)
        {
            Matrix x = eval.GetRow(row);
            Matrix y = neuralNetwork.Forward(x, true);
            Console.WriteLine($"x: {x.Array.GetValue(0, 0)}, {x.Array.GetValue(0, 1)}, p: {y.Array.GetValue(0, 0)}");
        }

        Console.ReadLine();
    }
}
