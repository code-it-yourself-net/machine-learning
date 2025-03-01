// Machine Learning
// File name: Program.cs
// Code It Yourself with .NET, 2024

// MNIST - Modified National Institute of Standards and Technology database

using MachineLearning;
using MachineLearning.NeuralNetwork;
using MachineLearning.NeuralNetwork.DataSources;
using MachineLearning.NeuralNetwork.Layers;
using MachineLearning.NeuralNetwork.LearningRates;
using MachineLearning.NeuralNetwork.Losses;
using MachineLearning.NeuralNetwork.Operations;
using MachineLearning.NeuralNetwork.Optimizers;
using MachineLearning.NeuralNetwork.ParamInitializers;

using Microsoft.Extensions.Logging;

using Serilog;

using static System.Console;

namespace MnistTests;

internal class ProgramMatrix
{
    public static void Run()
    {
        // Create ILogger using Serilog
        Serilog.Core.Logger serilog = new LoggerConfiguration()
            .WriteTo.File("..\\..\\..\\Logs\\log-.txt", rollingInterval: RollingInterval.Day)
            .CreateLogger();

        Log.Logger = serilog;
        Log.Information("Logging started...");

        // Create a LoggerFactory and add Serilog
        ILoggerFactory loggerFactory = new LoggerFactory()
            .AddSerilog(serilog);

        ILogger<Trainer> logger = loggerFactory.CreateLogger<Trainer>();

        Matrix train = Matrix.LoadCsv(".\\Data\\mnist_train_small.csv");
        Matrix test = Matrix.LoadCsv(".\\Data\\mnist_test.csv");

        (Matrix xTrain, Matrix yTrain) = Split(train);
        (Matrix xTest, Matrix yTest) = Split(test);

        // Scale xTrain and xTest to mean 0, variance 1
        WriteLine("Scale data to mean 0...");

        float mean = xTrain.Mean();
        xTrain.AddInPlace(-mean);
        xTest.AddInPlace(-mean);

        WriteLine($"xTrain min: {xTrain.Min()}");
        WriteLine($"xTest min: {xTest.Min()}");
        WriteLine($"xTrain max: {xTrain.Max()}");
        WriteLine($"xTest max: {xTest.Max()}");

        WriteLine("\nScale data to variance 1...");

        float std = xTrain.Std();
        xTrain.DivideInPlace(std);
        xTest.DivideInPlace(std);

        WriteLine($"xTrain min: {xTrain.Min()}");
        WriteLine($"xTest min: {xTest.Min()}");
        WriteLine($"xTrain max: {xTrain.Max()}");
        WriteLine($"xTest max: {xTest.Max()}");

        SimpleDataSource dataSource = new(xTrain, yTrain, xTest, yTest);
        SeededRandom commonRandom = new(Program.RandomSeed);

        // RangeInitializer initializer = new(-1f, 1f);
        GlorotInitializer initializer = new(commonRandom);
        Dropout? dropout1 = new(0.85f, commonRandom);
        Dropout? dropout2 = new(0.85f, commonRandom);

        // Define the network.
        NeuralNetwork model = new(
            layers: [
                new DenseLayer(178, new Tanh(), initializer, dropout1),
                new DenseLayer(46, new Tanh(), initializer, dropout2),
                // TODO: Try to change Linear to Softmax, and SoftmaxCrossEntropyLoss to CrossEntropyLoss.
                new DenseLayer(10, new Linear(), initializer)
            ],
            lossFunction: new SoftmaxCrossEntropyLoss()
        );

        WriteLine("\nStart training...\n");

        LearningRate learningRate = new ExponentialDecayLearningRate(0.19f, 0.05f);
        Trainer trainer = new(model, new StochasticGradientDescentMomentum(learningRate, 0.9f), random: commonRandom, logger: logger)
        {
            Memo = $"MATRIX: batch={Program.BatchSize}, seed={Program.RandomSeed}, epochs={Program.Epochs}."
        };

        trainer.Fit(
            dataSource, 
            EvalFunction, 
            epochs: Program.Epochs, 
            evalEveryEpochs: 1, 
            batchSize: Program.BatchSize
        );

        ReadLine();
    }

    private static float EvalFunction(NeuralNetwork neuralNetwork, Matrix xEvalTest, Matrix yEvalTest)
    {
        Matrix prediction = neuralNetwork.Forward(xEvalTest, true);
        Matrix predictionArgmax = prediction.Argmax();

        int rows = predictionArgmax.Array.GetLength(0);

#if DEBUG
        if (rows != yEvalTest.GetDimension(Dimension.Rows))
        {
            throw new ArgumentException("Number of samples in prediction and yEvalTest do not match.");
        }
#endif

        int hits = 0;
        for (int row = 0; row < rows; row++)
        {
            int predictedDigit = Convert.ToInt32(predictionArgmax.Array[row, 0]);
            if (yEvalTest.Array[row, predictedDigit] == 1f)
                hits++;
        }

        float accuracy = (float)hits / rows;
        return accuracy;
    }

    private static (Matrix xTest, Matrix yTest) Split(Matrix source)
    {
        // Split into xTest (all columns except the first one) and yTest (a one-hot table from the first column with values from 0 to 9).

        Matrix xTest = source.GetColumns(1..source.GetDimension(Dimension.Columns));
        Matrix yTest = source.GetColumn(0);

        // Convert yTest to a one-hot table.
        Matrix oneHot = new(yTest.GetDimension(Dimension.Rows), 10);
        for (int row = 0; row < yTest.GetDimension(Dimension.Rows); row++)
        {
            int value = Convert.ToInt32(yTest.Array[row, 0]);
            oneHot.Array[row, value] = 1f;
        }

        return (xTest, oneHot);
    }
}
