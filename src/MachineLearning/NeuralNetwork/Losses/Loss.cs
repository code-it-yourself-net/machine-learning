// Machine Learning Utils
// File name: Loss.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using MachineLearning.NeuralNetwork.Exceptions;

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork.Losses;

/// <summary>
/// The "loss" of a neural network.
/// </summary>
public abstract class Loss
{
    private Matrix? _prediction;
    private Matrix? _target;

    internal protected Matrix Prediction => _prediction ?? throw new NotYetCalculatedException();
    internal protected Matrix Target => _target ?? throw new NotYetCalculatedException();

    /// <summary>
    /// Computes the actual loss value
    /// </summary>
    public float Forward(Matrix prediction, Matrix target)
    {
        EnsureSameShape(prediction, target);
        _prediction = prediction;
        _target = target;

        return CalculateLoss();
    }

    /// <summary>
    /// Computes gradient of the loss value with respect to the input to the loss function.
    /// </summary>
    public Matrix Backward()
    {
        Matrix lossGradient = CalculateLossGradient();
        EnsureSameShape(_prediction, lossGradient);
        return lossGradient;
    }

    protected abstract float CalculateLoss();

    protected abstract Matrix CalculateLossGradient();

    #region Clone

    protected virtual Loss CloneBase()
    {
        Loss clone =(Loss)MemberwiseClone();
        clone._prediction = _prediction?.Clone();
        clone._target = _target?.Clone();
        return clone;
    }

    public Loss Clone() => CloneBase();

    #endregion
}
