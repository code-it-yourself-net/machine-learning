// Machine Learning
// File name: Program.cs
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

using Microsoft.Extensions.Logging;

using Serilog;

// Create ILogger using Serilog
Serilog.Core.Logger serilog = new LoggerConfiguration()
    .WriteTo.Console()
    // The folder path is set to src\LoggerTest\Logs. Change it according to your project structure.
    .WriteTo.File("..\\..\\..\\Logs\\log-.txt", rollingInterval: RollingInterval.Day)
    .CreateLogger();

Log.Logger = serilog;
Log.Information("Logging started...");

// Create a LoggerFactory and add Serilog
ILoggerFactory loggerFactory = new LoggerFactory()
    .AddSerilog(serilog);

try
{
    float[,] arguments = new float[,] { { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 }, { 0, 0 }, { -2, -6 }, { -3, -2 }, { 1, 6 }, { -3, 0 }, { 3, 0 }, { -2, 0 }, { -2.5f, 1 }, { -4, 5 }, { -5, 7 }, { -1, 3 }, { -3, 1 }, { -.5f, 8 }, { -1, 4 }, { .5f, 7 }, { 5, .1f }, { 1, -1 }, { 2, -3 }, { -4, -4 }, { 4, 4 }, { -1, -1 } };

    // y = 2 * x1 ^ 2 - 6 * x2 + 4.5
    FunctionDataSource dataSource = new(arguments, (float[] x) => 2f * x[0] * x[0] - 6f * x[1] + 4.5f, 0);

    ParamInitializer rangeInitializer = new RangeInitializer(-.5f, .5f);

    NeuralNetwork neuralNetwork = new(
        layers: [
            new DenseLayer(4, new Sigmoid(), rangeInitializer),
                new DenseLayer(4, new Sigmoid(), rangeInitializer),
                new DenseLayer(4, new Sigmoid(), rangeInitializer),
                new DenseLayer(1, new Linear(), rangeInitializer)
        ],
        lossFunction: new MeanSquaredError()
    );

    // Create an ILogger instance
    ILogger<Trainer> logger = loggerFactory.CreateLogger<Trainer>();

    LearningRate learningRate = new ConstantLearningRate(0.002f);
    Trainer trainer = new(neuralNetwork, new StochasticGradientDescent(learningRate), ConsoleOutputMode.Disable, null, logger)
    {
        Memo = "Logger test"
    };
    trainer.Fit(dataSource, batchSize: 32, epochs: 1_000, evalEveryEpochs: 1_000);
    Console.WriteLine("Training completed.");
}
finally
{
    Log.CloseAndFlush();
}
Console.ReadLine();
