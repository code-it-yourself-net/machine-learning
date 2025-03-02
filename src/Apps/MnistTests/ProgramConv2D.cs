// Machine Learning
// File name: ProgramConv2D.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

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

internal class MnistConvNeuralNetwork(SeededRandom? random)
    : NeuralNetwork<float[,,,], float[,]>(new SoftmaxCrossEntropyLoss(), random)
{
    protected override LayerListBuilder<float[,,,], float[,]> CreateLayerListBuilder()
    {
        ParamInitializer initializer = new GlorotInitializer(Random);
        // ParamInitializer initializer = new RangeInitializer(1, 1);
        Dropout4D? dropout = new(0.85f, Random);

        return AddLayer(new Conv2DLayer(
                filters: 32, // 16,
                kernelSize: 3,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer,
                dropout: dropout
            ))
            .AddLayer(new FlattenLayer())
            .AddLayer(new DenseLayer(10, new Linear(), initializer));
    }

}

internal class ProgramConv2D
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

        ILogger<Trainer4D> logger = loggerFactory.CreateLogger<Trainer4D>();

        // rows - batch
        // cols - features
        float[,] train = LoadCsv(".\\Data\\mnist_train_small.csv");
        float[,] test = LoadCsv(".\\Data\\mnist_test.csv");

        (float[,,,] xTrain, float[,] yTrain) = Split(train);
        (float[,,,] xTest, float[,] yTest) = Split(test);

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

        SimpleDataSource<float[,,,], float[,]> dataSource = new(xTrain, yTrain, xTest, yTest);

        SeededRandom commonRandom = new(Program.RandomSeed);

        // Declare the network.
        MnistConvNeuralNetwork model = new(commonRandom);

        WriteLine("\nStart training (Convolution2D)...\n");

        LearningRate learningRate = new ExponentialDecayLearningRate(0.01f, 0.001f);
        Optimizer optimizer = new StochasticGradientDescentMomentum(learningRate, 0.9f);
        //Optimizer optimizer = new StochasticGradientDescent(learningRate);
        Trainer4D trainer = new(model, optimizer, random: commonRandom, logger: logger)
        {
            Memo = $"CONV2D: batch={Program.BatchSize}, seed={Program.RandomSeed}, epochs={Program.Epochs}."
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

    private static float EvalFunction(NeuralNetwork<float[,,,], float[,]> neuralNetwork, float[,,,] xEvalTest, float[,] yEvalTest)
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

    private static (float[,,,] xTest, float[,] yTest) Split(float[,] source)
    {
        // Split into xTest (all columns except the first one) and yTest (a one-hot table from the first column with values from 0 to 9).

        float[,] xTest2D = source.GetColumns(1..source.GetLength((int)Dimension.Columns));
        float[,] yTest = source.GetColumn(0);

        Debug.Assert(xTest2D.GetLength(1) == 28 * 28);

        // Convert yTest to a one-hot table.
        int yTestRows = yTest.GetLength((int)Dimension.Rows);
        float[,] oneHot = new float[yTestRows, 10];
        for (int row = 0; row < yTestRows; row++)
        {
            int value = Convert.ToInt32(yTest[row, 0]);
            oneHot[row, value] = 1f;
        }

        int xTestRows = xTest2D.GetLength((int)Dimension.Rows);
        int xTestCols = xTest2D.GetLength((int)Dimension.Columns);
        float[,,,] xTest4D = new float[xTestRows, 1, 28, 28];

        for (int row = 0; row < xTestRows; row++)
        {
            for (int col = 0; col < xTestCols; col++)
            {
                //int x = col % 28;
                //int y = col / 28;
                xTest4D[row, 0, col / 28, col % 28] = xTest2D[row, col];
            }
        }

        return (xTest4D, oneHot);
    }
}
