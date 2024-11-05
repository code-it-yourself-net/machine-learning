// Machine Learning Utils
// File name: RangeInitializer.cs
// Code It Yourself with .NET, 2024

using static MachineLearning.Typed.ArrayUtils;

namespace MachineLearning.Typed.NeuralNetwork.ParamInitializers;

public class RangeInitializer(float from, float to) : ParamInitializer
{
    internal override float[] InitBiases(int neurons) 
        => CreateZeros(neurons);

    internal override float[,] InitWeights(int inputColumns, int neurons) 
        => CreateRange(inputColumns, neurons, from, to);

    public override string ToString() => $"RangeInitializer (from={from}, to={to})";
}
