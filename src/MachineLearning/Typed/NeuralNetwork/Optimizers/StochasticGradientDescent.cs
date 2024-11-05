// Machine Learning Utils
// File name: StochasticGradientDescent.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

using MachineLearning.NeuralNetwork.LearningRates;
using MachineLearning.Typed.NeuralNetwork.Layers;

namespace MachineLearning.Typed.NeuralNetwork.Optimizers;

public class StochasticGradientDescent(LearningRate learningRate) : Optimizer(learningRate)
{
//    public override void Step(NeuralNetwork neuralNetwork)
//    {
//        Matrix[] @params = neuralNetwork.GetAllParams();
//        Matrix[] paramGrads = neuralNetwork.GetAllParamGradients();

//#if DEBUG
//        if (@params.Length != paramGrads.Length)
//        {
//            throw new ArgumentException("Number of parameters and gradients do not match.");
//        }
//#endif

//        float learningRate = LearningRate.GetLearningRate();

//        // Iterate through both lists in parallel
//        for (int i = 0; i < @params.Length; i++)
//        {
//            Matrix param = @params[i];
//            Matrix paramGrad = paramGrads[i];

//            // Update the parameter
//            Matrix deltaParamGrad = paramGrad.Multiply(learningRate);
//            param.SubtractInPlace(deltaParamGrad);
//        }
//    }

    public override void Update(Layer layer, float[] param, float[] paramGradient) 
    { 
        Debug.Assert(param.GetLength(0) == paramGradient.GetLength(0));

        float learningRate = LearningRate.GetLearningRate();

        for (int row = 0; row < param.Length; row++)
        {
            param[row] -= learningRate * paramGradient[row];
        }
    }

    public override void Update(Layer layer, float[,] param, float[,] paramGradient)
    {
        Debug.Assert(param.GetLength(0) == paramGradient.GetLength(0));
        Debug.Assert(param.GetLength(1) == paramGradient.GetLength(1));

        float learningRate = LearningRate.GetLearningRate();

        for (int row = 0; row < param.GetLength(0); row++)
        {
            for (int col = 0; col < param.GetLength(1); col++)
            {
                param[row, col] -= learningRate * paramGradient[row, col];
            }
        }
    }

    public override string ToString() => $"StochasticGradientDescent (learningRate={LearningRate})";
}
