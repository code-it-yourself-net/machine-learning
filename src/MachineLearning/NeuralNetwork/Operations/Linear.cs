// Machine Learning Utils
// File name: Linear.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// "Identity" activation function
/// </summary>
public class Linear : Operation
{
    protected override Matrix CalcOutput(bool inference) => Input;

    protected override Matrix CalcInputGradient(Matrix outputGradient) => outputGradient;

    public override string ToString() => "Linear";
}
