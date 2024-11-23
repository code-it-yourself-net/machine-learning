// Machine Learning Utils
// File name: StochasticGradientDescent.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

using MachineLearning.NeuralNetwork.LearningRates;
using MachineLearning.Typed.NeuralNetwork.Layers;

namespace MachineLearning.Typed.NeuralNetwork.Optimizers;

public class StochasticGradientDescent(LearningRate learningRate) : Optimizer(learningRate)
{
    public override void Update(Layer layer, float[] param, float[] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();

        for (int i = 0; i < param.Length; i++)
        {
            param[i] -= learningRate * paramGradient[i];
        }
    }

    public override void Update(Layer layer, float[,] param, float[,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();

        for (int i = 0; i < param.GetLength(0); i++)
        {
            for (int j = 0; j < param.GetLength(1); j++)
            {
                param[i, j] -= learningRate * paramGradient[i, j];
            }
        }
    }

    public override void Update(Layer layer, float[,,,] param, float[,,,] paramGradient)
    {
        Debug.Assert(param.HasSameShape(paramGradient));

        float learningRate = LearningRate.GetLearningRate();

        for (int i = 0; i < param.GetLength(0); i++)
        {
            for (int j = 0; j < param.GetLength(1); j++)
            {
                for (int k = 0; k < param.GetLength(2); k++)
                {
                    for (int l = 0; l < param.GetLength(3); l++)
                    {
                        param[i, j, k, l] -= learningRate * paramGradient[i, j, k, l];
                    }
                }
            }
        }
    }

    public override string ToString() => $"StochasticGradientDescent (learningRate={LearningRate})";

}
