// Machine Learning Utils
// File name: GlorotInitializer.cs
// Code It Yourself with .NET, 2024

using static MachineLearning.Typed.ArrayUtils;

namespace MachineLearning.Typed.NeuralNetwork.ParamInitializers;

public class GlorotInitializer(SeededRandom? random = null) : RandomInitializer(random)
{
    internal override float[,] InitWeights(int inputColumns, int neurons)
    {
        float stdDev = (float)Math.Sqrt(2.0 / (inputColumns + neurons));
        return CreateRandomNormal(inputColumns, neurons, Random, 0, stdDev);
    }

    internal override float[] InitBiases(int neurons) 
        => CreateZeros(neurons);

    public override string ToString() 
        => $"GlorotInitializer (seed={Seed})";
}