// Machine Learning
// File name: LinearRegression.cs
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

internal class LinearRegression
{
    private static void Main(string[] args)
    {
        float[,] arguments = new float[,] { { 10, 20 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 }, { 0, 0 }, { -5, -6 }, { -100f, 2f } };

        // y = -(x1 + x2)
        FunctionDataSource dataSource = new(arguments, (float[] x) => -x[0] - x[1], 0);
        SeededRandom random = new(12345);
        NeuralNetwork linearRegression = new(
            layers: [new DenseLayer(1, new Linear(), new RandomInitializer(random))],
            lossFunction: new MeanSquaredError()
        );

        LearningRate learningRate = new ConstantLearningRate(0.00001f);
        Trainer trainer = new(linearRegression, new StochasticGradientDescent(learningRate), ConsoleOutputMode.OnlyOnEval);
        trainer.Fit(dataSource, batchSize: 32, epochs: 1_500, evalEveryEpochs: 100);

        Matrix[] @params = linearRegression.GetAllParams();
        Matrix weights = @params[0];
        Matrix bias = @params[1];
        float loss = linearRegression.LastLoss;

        Console.WriteLine();
        Console.WriteLine($"weights: \n{weights}");
        Console.WriteLine($"bias: {bias}");
        Console.WriteLine($"loss: {loss}");
        Console.WriteLine();

        Console.ReadLine();
    }
}
