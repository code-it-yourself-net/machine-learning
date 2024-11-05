// Machine Learning Utils
// File name: Trainer.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

using MachineLearning.NeuralNetwork.DataSources;
using MachineLearning.NeuralNetwork.Layers;
using MachineLearning.NeuralNetwork.Optimizers;

using Microsoft.Extensions.Logging;

using static System.Console;
using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork;

/// <summary>
/// Represents a trainer for a neural network.
/// </summary>
public class Trainer(
    NeuralNetwork neuralNetwork,
    Optimizer optimizer,
    ConsoleOutputMode consoleOutputMode = ConsoleOutputMode.OnlyOnEval,
    SeededRandom? random = null,
    ILogger<Trainer>? logger = null)
{
    private float _bestLoss = float.MaxValue;

    /// <summary>
    /// Gets or sets the memo associated with the trainer.
    /// </summary>
    public string? Memo { get; set; }

    /// <summary>
    /// Generates batches of input and output matrices.
    /// </summary>
    /// <param name="x">The input matrix.</param>
    /// <param name="y">The output matrix.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>An enumerable of batches.</returns>
    public static IEnumerable<(Matrix xBatch, Matrix yBatch)> GenerateBatches(Matrix x, Matrix y, int batchSize = 32)
    {
        int trainRows = x.GetDimension(Dimension.Rows);
#if DEBUG
        if (trainRows != y.GetDimension(Dimension.Rows))
        {
            throw new ArgumentException("Number of samples in x and y do not match.");
        }
#endif
        for (int batchStart = 0; batchStart < trainRows; batchStart += batchSize)
        {
            int effectiveBatchSize = Math.Min(batchSize, trainRows - batchStart);
            int batchEnd = effectiveBatchSize + batchStart;
            Matrix xBatch = x.GetRows(batchStart..batchEnd);
            Matrix yBatch = y.GetRows(batchStart..batchEnd);
            yield return (xBatch, yBatch);
        }
    }

    /// <summary>
    /// Fits the neural network to the provided data source.
    /// </summary>
    /// <param name="dataSource">The data source.</param>
    /// <param name="evalFunction">The evaluation function.</param>
    /// <param name="epochs">The number of epochs.</param>
    /// <param name="evalEveryEpochs">The number of epochs between evaluations.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="restart">A flag indicating whether to restart the training.</param>
    public void Fit(
        DataSource dataSource,
        Func<NeuralNetwork, Matrix, Matrix, float>? evalFunction = null,
        int epochs = 100,
        int evalEveryEpochs = 10,
        int batchSize = 32,
        bool earlyStop = false,
        bool restart = true)
    {
        Stopwatch trainWatch = Stopwatch.StartNew();

        logger?.LogInformation("Fit started with params: epochs: {epochs}, batchSize: {batchSize}, optimizer: {optimizer}, random: {random}.", epochs, batchSize, optimizer, random);
        logger?.LogInformation("Model layers:");
        foreach (Layer layer in neuralNetwork.Layers)
        {
            logger?.LogInformation("Layer: {layer}.", layer);
        }
        logger?.LogInformation("Loss function: {loss}", neuralNetwork.LossFunction);

        if (Memo is not null)
            logger?.LogInformation("Memo: \"{memo}\".", Memo);

#if DEBUG
        string environment = "Debug";
#else
        string environment = "Release";
#endif
        logger?.LogInformation("Environment: {environment}.", environment);

        (Matrix xTrain, Matrix yTrain, Matrix? xTest, Matrix? yTest) = dataSource.GetData();

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            logger?.LogInformation("Epoch {epoch}/{epochs} started.", epoch, epochs);

            bool evaluationEpoch = epoch % evalEveryEpochs == 0;
            bool eval = xTest is not null && yTest is not null && evaluationEpoch;

            if ((evaluationEpoch && consoleOutputMode == ConsoleOutputMode.OnlyOnEval) || consoleOutputMode == ConsoleOutputMode.Always)
                WriteLine($"Epoch {epoch}/{epochs}...");

            // Epoch should be later than 1 to save the first checkpoint.
            //if (eval && epoch > 1)
            //{
            //    neuralNetwork.SaveCheckpoint();
            //    logger?.LogInformation("Checkpoint saved.");
            //}

            (xTrain, yTrain) = PermuteData(xTrain, yTrain, random ?? new Random());
            optimizer.UpdateLearningRate(epoch, epochs);

            float? trainLoss = null;
            int step = 0;
            int allSteps = (int)Math.Ceiling(xTrain.GetDimension(Dimension.Rows) / (float)batchSize);
            float? stepsPerSecond = null;

            Stopwatch stepWatch = Stopwatch.StartNew();
            foreach ((Matrix xBatch, Matrix yBatch) in GenerateBatches(xTrain, yTrain, batchSize))
            {
                step++;
                if (allSteps > 1 && consoleOutputMode > ConsoleOutputMode.Disable)
                {
                    string stepInfo = $"Step {step}/{allSteps}...";
                    if (stepsPerSecond is not null)
                        stepInfo += $" {stepsPerSecond.Value:F2} steps/s";
                    Write(stepInfo + "\r");
                }

                trainLoss = (trainLoss ?? 0) + neuralNetwork.TrainBatch(xBatch, yBatch);
                optimizer.Step(neuralNetwork);

                long elapsedMsPerStep = stepWatch.ElapsedMilliseconds / step;
                stepsPerSecond = 1000.0f / elapsedMsPerStep;
            }
            stepWatch.Stop();

            if (trainLoss is not null && evaluationEpoch)
            {
                if (consoleOutputMode > ConsoleOutputMode.Disable)
                    WriteLine($"Train loss (average): {trainLoss.Value / allSteps}");
                logger?.LogInformation("Train loss (average): {trainLoss} for epoch {epoch}.", trainLoss.Value / allSteps, epoch);
            }

            if (eval)
            {
                Matrix testPredictions = neuralNetwork.Forward(xTest!, true);
                float loss = neuralNetwork.LossFunction.Forward(testPredictions, yTest!);

                if (consoleOutputMode > ConsoleOutputMode.Disable)
                    WriteLine($"Test loss: {loss}");
                logger?.LogInformation("Test loss: {testLoss} for epoch {epoch}.", loss, epoch);

                if (evalFunction is not null)
                {
                    float evalValue = evalFunction(neuralNetwork, xTest!, yTest!);

                    if (consoleOutputMode > ConsoleOutputMode.Disable)
                        WriteLine($"Eval: {evalValue:P2}");
                    logger?.LogInformation("Eval: {evalValue:P2} for epoch {epoch}.", evalValue, epoch);
                }

                if (loss < _bestLoss)
                {
                    _bestLoss = loss;
                }
                else if (earlyStop)
                {
                    if (neuralNetwork.HasCheckpoint())
                    {
                        neuralNetwork.RestoreCheckpoint();
                        logger?.LogInformation("Checkpoint restored.");
                    }

                    if (consoleOutputMode > ConsoleOutputMode.Disable)
                        WriteLine($"Early stopping, loss {loss} is greater than {_bestLoss}");
                    logger?.LogInformation("Early stopping. Loss {loss} is greater than {bestLoss}.", loss, _bestLoss);

                    break;
                }

            }
        }
        trainWatch.Stop();
        float elapsedSeconds = trainWatch.ElapsedMilliseconds / 1000.0f;
        logger?.LogInformation("Fit finished in {elapsedSecond:F2} s.", elapsedSeconds);
        if (consoleOutputMode > ConsoleOutputMode.Disable)
            WriteLine($"Fit finished in {elapsedSeconds:F2} s.");
    }
}
