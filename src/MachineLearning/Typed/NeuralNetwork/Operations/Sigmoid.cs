﻿// Machine Learning Utils
// File name: Sigmoid.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

namespace MachineLearning.Typed.NeuralNetwork.Operations;

/// <summary>
/// Sigmoid activation function.
/// </summary>
public class Sigmoid : Operation2D
{
    protected override float[,] CalcOutput(bool inference) => Input.Sigmoid();

    protected override float[,] CalcInputGradient(float[,] outputGradient)
    {
        // The CalcInputGradient function computes the gradient of the loss with respect to the input of the Sigmoid function.
        // This is done using the chain rule of calculus. Given the output gradient (dL/dy), the function calculates the input gradient (dL/dx).
        // The derivative of the Sigmoid function σ(x) = 1 / (1 + exp(-x)) is σ(x) * (1 - σ(x)).
        // Therefore, the input gradient is computed as: dL/dx = dL/dy * σ(x) * (1 - σ(x)).
        // The elementwise multiplication of the output gradient and the derivative of the Sigmoid function is returned as the input gradient.
        // σ(x) => Output
        // dL/dy => outputGradient
        // dl/dx => inputGradient
        float[,] sigmoidBackward = Output.MultiplyElementwise(Output.AsOnes().Subtract(Output));
        return outputGradient.MultiplyElementwise(sigmoidBackward);
    }

    public override string ToString() => "Sigmoid";
}
