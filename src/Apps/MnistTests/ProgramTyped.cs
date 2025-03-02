// Machine Learning
// File name: ProgramTyped.cs
// Code It Yourself with .NET, 2024

// MNIST - Modified National Institute of Standards and Technology database

using MachineLearning;
using MachineLearning.NeuralNetwork.LearningRates;
using MachineLearning.Typed;
using MachineLearning.Typed.NeuralNetwork;
using MachineLearning.Typed.NeuralNetwork.DataSources;
using MachineLearning.Typed.NeuralNetwork.Layers;
using MachineLearning.Typed.NeuralNetwork.Losses;
using MachineLearning.Typed.NeuralNetwork.Operations;
using MachineLearning.Typed.NeuralNetwork.Optimizers;
using MachineLearning.Typed.NeuralNetwork.ParamInitializers;

using Microsoft.Extensions.Logging;

using Serilog;

using static System.Console;
using static MachineLearning.Typed.ArrayUtils;

namespace MnistTests;

class MnistNeuralNetwork(SeededRandom? random) 
    : NeuralNetwork<float[,], float[,]>(new SoftmaxCrossEntropyLoss(), random)
{
    protected override LayerListBuilder<float[,], float[,]> CreateLayerListBuilder()
    {
        // RangeInitializer initializer = new(-1f, 1f);
        GlorotInitializer initializer = new(Random);
        Dropout2D? dropout1 = new(0.85f, Random);
        Dropout2D? dropout2 = new(0.85f, Random);

        return AddLayer(new DenseLayer(178, new Tanh2D(), initializer, dropout1))
            .AddLayer(new DenseLayer(46, new Tanh2D(), initializer, dropout2))
            .AddLayer(new DenseLayer(10, new Linear(), initializer));
    }

}

internal class ProgramTyped
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

        ILogger<Trainer2D> logger = loggerFactory.CreateLogger<Trainer2D>();

        // rows - batch
        // cols - features
        float[,] train = LoadCsv(".\\Data\\mnist_train_small.csv");
        float[,] test = LoadCsv(".\\Data\\mnist_test.csv");

        (float[,] xTrain, float[,] yTrain) = Split(train);
        (float[,] xTest, float[,] yTest) = Split(test);

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

        SimpleDataSource<float[,], float[,]> dataSource = new(xTrain, yTrain, xTest, yTest);

        SeededRandom commonRandom = new(Program.RandomSeed);

        // Declare the network.
        MnistNeuralNetwork model = new(commonRandom);

        WriteLine("\nStart training...\n");

        LearningRate learningRate = new ExponentialDecayLearningRate(0.19f, 0.05f);
        Trainer2D trainer = new(model, new StochasticGradientDescentMomentum(learningRate, 0.9f), random: commonRandom, logger: logger)
        {
            Memo = $"TYPED: batch={Program.BatchSize}, seed={Program.RandomSeed}, epochs={Program.Epochs}."
        };

        trainer.Fit(
            dataSource, 
            EvalFunction, 
            epochs: Program.Epochs, 
            evalEveryEpochs: 2, 
            logEveryEpochs: 1, 
            batchSize: Program.BatchSize
        );

        ReadLine();
    }

    private static float EvalFunction(NeuralNetwork<float[,], float[,]> neuralNetwork, float[,] xEvalTest, float[,] yEvalTest)
    {
        // 'prediction' is a one-hot table with the predicted digit.
        float[,] prediction = neuralNetwork.Forward(xEvalTest, true);
        int[] predictionArgmax = prediction.Argmax();

        int rows = predictionArgmax.GetLength(0);

#if DEBUG
        if (rows != yEvalTest.GetLength((int)Dimension.Rows))
        {
            throw new ArgumentException("Number of samples in prediction and yEvalTest do not match.");
        }
#endif

        int hits = 0;
        for (int row = 0; row < rows; row++)
        {
            int predictedDigit = predictionArgmax[row];
            if (yEvalTest[row, predictedDigit] == 1f)
                hits++;
        }

        float accuracy = (float)hits / rows;
        return accuracy;
    }

    private static (float[,] xTest, float[,] yTest) Split(float[,] source)
    {
        // Split into xTest (all columns except the first one) and yTest (a one-hot table from the first column with values from 0 to 9).

        float[,] xTest = source.GetColumns(1..source.GetLength((int)Dimension.Columns));
        float[,] yTest = source.GetColumn(0);

        // Convert yTest to a one-hot table.
        float[,] oneHot = new float[yTest.GetLength((int)Dimension.Rows), 10];
        for (int row = 0; row < yTest.GetLength((int)Dimension.Rows); row++)
        {
            int value = Convert.ToInt32(yTest[row, 0]);
            oneHot[row, value] = 1f;
        }

        return (xTest, oneHot);
    }
}
