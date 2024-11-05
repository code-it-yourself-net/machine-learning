// Machine Learning Utils
// File name: WeightMultiply.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// Weight multiplication operation for a neural network.
/// </summary>
/// <param name="weight">Weight matrix.</param>
public class WeightMultiply(Matrix weight) : ParamOperation(weight)
{
    protected override Matrix CalcOutput(bool inference)
        => Input.MultiplyDot(Param);

    protected override Matrix CalcInputGradient(Matrix outputGradient)
        => outputGradient.MultiplyDot(Param.Transpose());

    protected override Matrix CalcParamGradient(Matrix outputGradient)
        => Input.Transpose().MultiplyDot(outputGradient);
}
