// Machine Learning Utils
// File name: Optimizer.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using MachineLearning.NeuralNetwork.LearningRates;

namespace MachineLearning.NeuralNetwork.Optimizers;

/// <summary>
/// Base class for a neural network optimizer.
/// </summary>
public abstract class Optimizer(LearningRate learningRate)
{
    protected LearningRate LearningRate => learningRate;

    public abstract void Step(NeuralNetwork neuralNetwork);

    public virtual void UpdateLearningRate(int epoch, int epochs) => learningRate.Update(epoch, epochs);
}
