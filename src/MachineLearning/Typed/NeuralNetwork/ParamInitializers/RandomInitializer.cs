﻿// Machine Learning Utils
// File name: RandomInitializer.cs
// Code It Yourself with .NET, 2024

using static MachineLearning.Typed.ArrayUtils;

namespace MachineLearning.Typed.NeuralNetwork.ParamInitializers;

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

    internal override float[] InitBiases(int neurons) 
        => CreateRandom(neurons, _random);

    internal override float[,] InitWeights(int inputColumns, int neurons) 
        => CreateRandom(inputColumns, neurons, _random);

    public override string ToString() => $"RandomInitializer (seed={_seed})";
}
