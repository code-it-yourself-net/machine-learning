﻿// Machine Learning Utils
// File name: RandomInitializer.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.ParamInitializers;

public class RandomInitializer : ParamInitializer
{
    private readonly Random _random;
    private readonly int? _seed;

    public RandomInitializer(SeededRandom? random = null)
    {
        if(random != null)
        {
            _random = random;
            _seed = random.Seed;
        }
        else
        {
            _random = new Random();
            _seed = null;
        }
    }

    protected Random Random => _random;

    protected int? Seed => _seed;

    internal override Matrix InitBiases(int neurons) 
        => Matrix.Random(1, neurons, _random);

    internal override Matrix InitWeights(int inputColumns, int neurons) 
        => Matrix.Random(inputColumns, neurons, _random);

    public override string ToString() => $"RandomInitializer (seed={_seed})";
}
