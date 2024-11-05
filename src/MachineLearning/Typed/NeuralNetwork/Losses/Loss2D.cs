// Machine Learning Utils
// File name: Loss2D.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.Typed.NeuralNetwork.Losses;

public abstract class Loss2D : Loss<float[,]>
{
    protected override void EnsureSameShape(float[,]? prediction, float[,] target)
        => ArrayUtils.EnsureSameShape(prediction, target);
}
