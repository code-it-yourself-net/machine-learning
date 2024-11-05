// Machine Learning Utils
// File name: StochasticGradientDescentMomentum.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.LearningRates;

namespace MachineLearning.NeuralNetwork.Optimizers;

public class StochasticGradientDescentMomentum(LearningRate learningRate, float momentum) : Optimizer(learningRate)
{
    private Matrix[]? _velocities;

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

        if (_velocities == null)
        {
            // First step - create a new list of velocities for each parameter matrix.
            // It will be used in the next steps.
            _velocities = new Matrix[@params.Length];
            for (int i = 0; i < @params.Length; i++)
            {
                _velocities[i] = Matrix.Zeros(@params[i]);
            }
        }

        // Iterate through both lists in parallel
        for (int i = 0; i < @params.Length; i++)
        {
            Matrix param = @params[i];
            Matrix paramGrad = paramGrads[i];
            Matrix velocity = _velocities[i];

            // Update the velocity
            velocity.MultiplyInPlace(momentum);
            Matrix deltaParamGrad = paramGrad.Multiply(LearningRate.GetLearningRate());
            velocity.AddInPlace(deltaParamGrad);

            // Update the parameter
            param.SubtractInPlace(velocity);
        }
    }

    public override string ToString() => $"StochasticGradientDescentMomentum (learningRate={LearningRate}, momentum={momentum})";
}