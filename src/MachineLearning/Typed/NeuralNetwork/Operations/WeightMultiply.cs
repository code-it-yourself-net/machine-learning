// Machine Learning Utils
// File name: WeightMultiply.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using MachineLearning.Typed.NeuralNetwork.Layers;
using MachineLearning.Typed.NeuralNetwork.Optimizers;

using static MachineLearning.Typed.ArrayUtils;

namespace MachineLearning.Typed.NeuralNetwork.Operations;

/// <summary>
/// Weight multiplication operation for a neural network.
/// </summary>
/// <param name="weights">Weight matrix.</param>
public class WeightMultiply(float[,] weights) : ParamOperation2D<float[,]>(weights)
{
    protected override float[,] CalcOutput(bool inference)
        => Input.MultiplyDot(Param);

    protected override float[,] CalcInputGradient(float[,] outputGradient)
        => outputGradient.MultiplyDot(Param.Transpose());

    protected override float[,] CalcParamGradient(float[,] outputGradient)
        => Input.Transpose().MultiplyDot(outputGradient);

    public override void UpdateParams(Layer layer, Optimizer optimizer)
    {
        optimizer.Update(layer, Param, ParamGradient);
    }

    protected override void EnsureSameShapeForParam(float[,]? param, float[,] paramGradient) 
        => EnsureSameShape(param, paramGradient);

    public override int GetParamCount()
        => Param.Length;
}
