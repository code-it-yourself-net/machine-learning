// Machine Learning Utils
// File name: StochasticGradientDescentMomentum.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;
using System.Runtime.CompilerServices;

using MachineLearning.NeuralNetwork.LearningRates;
using MachineLearning.Typed.NeuralNetwork.Layers;

namespace MachineLearning.Typed.NeuralNetwork.Optimizers;

public class StochasticGradientDescentMomentum(LearningRate learningRate, float momentum) : Optimizer(learningRate)
{
    //private Matrix[]? _velocities;

    private Dictionary<float[], float[]> _velocities1D = new();
    private Dictionary<float[,], float[,]> _velocities2D = new();

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

    //        if (_velocities == null)
    //        {
    //            // First step - create a new list of velocities for each parameter matrix.
    //            // It will be used in the next steps.
    //            _velocities = new Matrix[@params.Length];
    //            for (int i = 0; i < @params.Length; i++)
    //            {
    //                _velocities[i] = Matrix.Zeros(@params[i]);
    //            }
    //        }

    //        float learningRate = LearningRate.GetLearningRate();

    //        // Iterate through both lists in parallel
    //        for (int i = 0; i < @params.Length; i++)
    //        {
    //            Matrix param = @params[i];
    //            Matrix paramGrad = paramGrads[i];
    //            Matrix velocity = _velocities[i];

    //            // Update the velocity
    //            velocity.MultiplyInPlace(momentum);
    //            Matrix deltaParamGrad = paramGrad.Multiply(learningRate);
    //            velocity.AddInPlace(deltaParamGrad);

    //            // Update the parameter
    //            param.SubtractInPlace(velocity);
    //        }
    //    }

    public override string ToString() => $"StochasticGradientDescentMomentum (learningRate={LearningRate}, momentum={momentum})";

    public override void Update(Layer layer, float[] param, float[] paramGradient)
    {
        Debug.Assert(param.GetLength(0) == paramGradient.GetLength(0));

        float learningRate = LearningRate.GetLearningRate();

        float[] velocities = GetOrCreateVelocities(param);

        for (int row = 0; row < param.Length; row++)
        {
            velocities[row] = velocities[row] * momentum + learningRate * paramGradient[row];
            param[row] -= velocities[row];
        }
    }

    public override void Update(Layer layer, float[,] param, float[,] paramGradient)
    {
        Debug.Assert(param.GetLength(0) == paramGradient.GetLength(0));
        Debug.Assert(param.GetLength(1) == paramGradient.GetLength(1));

        float learningRate = LearningRate.GetLearningRate();

        float[,] velocities = GetOrCreateVelocities(param);

        for (int row = 0; row < param.GetLength(0); row++)
        {
            for (int col = 0; col < param.GetLength(1); col++)
            {
                velocities[row, col] = velocities[row, col] * momentum + learningRate * paramGradient[row, col];
                param[row, col] -= velocities[row, col];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float[] GetOrCreateVelocities(float[] param)
    {
        if(_velocities1D.TryGetValue(param, out float[]? velocities))
        {
            return velocities;
        }
        else
        {
            velocities = new float[param.Length];
            _velocities1D.Add(param, velocities);
            return velocities;
        }

        //if (_velocities1D.ContainsKey(param))
        //{
        //    return _velocities1D[param];
        //}
        //else
        //{
        //    float[] velocities = new float[param.Length];
        //    _velocities1D.Add(param, velocities);
        //    return velocities;
        //}
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float[,] GetOrCreateVelocities(float[,] param)
    {
        if(_velocities2D.TryGetValue(param, out float[,]? velocities))
        {
            return velocities;
        }
        else
        {
            velocities = new float[param.GetLength(0), param.GetLength(1)];
            _velocities2D.Add(param, velocities);
            return velocities;
        }

        //if (_velocities2D.ContainsKey(param))
        //{
        //    return _velocities2D[param];
        //}
        //else
        //{
        //    float[,] velocities = new float[param.GetLength(0), param.GetLength(1)];
        //    _velocities2D.Add(param, velocities);
        //    return velocities;
        //}
    }
}