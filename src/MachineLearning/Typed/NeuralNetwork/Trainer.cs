﻿// Machine Learning Utils
// File name: Trainer.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

using MachineLearning.Typed.NeuralNetwork.DataSources;
using MachineLearning.Typed.NeuralNetwork.Layers;
using MachineLearning.Typed.NeuralNetwork.Optimizers;

using Microsoft.Extensions.Logging;

using static System.Console;

namespace MachineLearning.Typed.NeuralNetwork;

/// <summary>
/// Represents a trainer for a neural network.
/// </summary>
public abstract class Trainer<TInputData, TPrediction>(
    NeuralNetwork<TInputData, TPrediction> neuralNetwork,
    Optimizer optimizer,
    ConsoleOutputMode consoleOutputMode = ConsoleOutputMode.OnlyOnEval,
    SeededRandom? random = null,
    ILogger<Trainer<TInputData, TPrediction>>? logger = null)
    where TInputData : notnull
    where TPrediction : notnull
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
    protected abstract IEnumerable<(TInputData xBatch, TPrediction yBatch)> GenerateBatches(TInputData x, TPrediction y, int batchSize = 32);

    protected abstract (TInputData, TPrediction) PermuteData(TInputData x, TPrediction y, Random random);

    protected abstract float GetRows(TInputData x);

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
        DataSource<TInputData, TPrediction> dataSource,
        Func<NeuralNetwork<TInputData, TPrediction>, TInputData, TPrediction, float>? evalFunction = null,
        int epochs = 100,
        int evalEveryEpochs = 10,
        int logEveryEpochs = 1,
        int batchSize = 32,
        bool earlyStop = false,
        bool restart = true)
    {
        Stopwatch trainWatch = Stopwatch.StartNew();

        logger?.LogInformation("");
        logger?.LogInformation("===== Begin Log =====");
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

        (TInputData xTrain, TPrediction yTrain, TInputData? xTest, TPrediction? yTest) = dataSource.GetData();
        int allSteps = (int)Math.Ceiling(GetRows(xTrain) / (float)batchSize);

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            logger?.LogInformation("Epoch {epoch}/{epochs} started.", epoch, epochs);

            bool lastEpoch = epoch == epochs;
            bool evaluationEpoch = epoch % evalEveryEpochs == 0 || lastEpoch;
            bool logEpoch = epoch % logEveryEpochs == 0 || lastEpoch;

            bool eval = xTest is not null && yTest is not null && evaluationEpoch;

            if ((logEpoch && consoleOutputMode == ConsoleOutputMode.OnlyOnEval) || consoleOutputMode == ConsoleOutputMode.Always)
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

            float? stepsPerSecond = null;

            Stopwatch stepWatch = Stopwatch.StartNew();
            foreach ((TInputData xBatch, TPrediction yBatch) in GenerateBatches(xTrain, yTrain, batchSize))
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
                //optimizer.Step(neuralNetwork);
                neuralNetwork.UpdateParams(optimizer);

                long elapsedMsPerStep = stepWatch.ElapsedMilliseconds / step;
                stepsPerSecond = 1000.0f / elapsedMsPerStep;
            }
            stepWatch.Stop();

            // Write a line with 80 spaces to clean the line with the step info.
            if (allSteps > 1 && consoleOutputMode > ConsoleOutputMode.Disable)
                Write(new string(' ', 80) + "\r");

            if (trainLoss is not null && logEpoch)
            {
                if (consoleOutputMode > ConsoleOutputMode.Disable)
                    WriteLine($"Train loss (average): {trainLoss.Value / allSteps}");
                logger?.LogInformation("Train loss (average): {trainLoss} for epoch {epoch}.", trainLoss.Value / allSteps, epoch);
            }

            if (eval)
            {
                TPrediction testPredictions = neuralNetwork.Forward(xTest!, true);
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

        int paramCount = neuralNetwork.GetParamCount();
        logger?.LogInformation("{paramCount:n0} parameters trained.", paramCount);
        if (consoleOutputMode > ConsoleOutputMode.Disable)
            WriteLine($"{paramCount:n0} parameters trained.");

        logger?.LogInformation("===== End Log =====");
        
    }


}
