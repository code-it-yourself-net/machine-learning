// Machine Learning Utils
// File name: ParamInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.Typed.NeuralNetwork.ParamInitializers;

public abstract class ParamInitializer
{
    internal abstract float[] InitBiases(int neurons);
    internal abstract float[,] InitWeights(int inputColumns, int neurons);
}
