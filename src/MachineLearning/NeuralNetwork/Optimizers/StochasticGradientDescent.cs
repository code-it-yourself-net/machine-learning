// Machine Learning Utils
// File name: StochasticGradientDescent.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.LearningRates;

namespace MachineLearning.NeuralNetwork.Optimizers;

public class StochasticGradientDescent(LearningRate learningRate) : Optimizer(learningRate)
{
    public override void Step(NeuralNetwork neuralNetwork)
    {
        Matrix[] @params = neuralNetwork.GetAllParams();
        Matrix[] paramGrads = neuralNetwork.GetAllParamGradients();

#if DEBUG
        if (@params.Length != paramGrads.Length)
        {
            throw new ArgumentException("Number of parameters and gradients do not match.");
        }
#endif

        // Iterate through both lists in parallel
        for (int i = 0; i < @params.Length; i++)
        {
            Matrix param = @params[i];
            Matrix paramGrad = paramGrads[i];

            // Update the parameter
            Matrix deltaParamGrad = paramGrad.Multiply(LearningRate.GetLearningRate());
            param.SubtractInPlace(deltaParamGrad);
        }
    }

    public override string ToString() => $"StochasticGradientDescent (learningRate={LearningRate})";
}
