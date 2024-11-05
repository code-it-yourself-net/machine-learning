// Machine Learning Utils
// File name: Operation.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using MachineLearning.NeuralNetwork.Exceptions;

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// Base class for an "operation" in a neural network.
/// </summary>
public abstract class Operation
{
    private Matrix? _input;
    // private Matrix? _inputGradient; // not used - to remove
    private Matrix? _output;

    protected Matrix Input => _input ?? throw new NotYetCalculatedException();

    protected Matrix Output => _output ?? throw new NotYetCalculatedException();

    /// <summary>
    /// Converts input to output.
    /// </summary>
    public virtual Matrix Forward(Matrix input, bool inference)
    {
        _input = input;
        _output = CalcOutput(inference);
        return _output;
    }

    /// <summary>
    /// Converts output gradient to input gradient.
    /// </summary>
    public virtual Matrix Backward(Matrix outputGradient)
    {
        EnsureSameShape(_output, outputGradient);
        Matrix? inputGradient = CalcInputGradient(outputGradient);

        EnsureSameShape(_input, inputGradient);
        return inputGradient;
    }

    /// <summary>
    /// Computes output.
    /// </summary>
    protected abstract Matrix CalcOutput(bool inference);

    /// <summary>
    /// Calculates input gradient.
    /// </summary>
    /// <remarks>
    /// Na podstawie outputGradient oblicza zmiany w input.
    /// </remarks>
    protected abstract Matrix CalcInputGradient(Matrix outputGradient);

    #region Clone

    protected virtual Operation CloneBase()
    {
        Operation clone =(Operation)MemberwiseClone();
        clone._input = _input?.Clone();
        //clone._inputGradient = _inputGradient?.Clone();
        clone._output = _output?.Clone();
        return clone;
    }

    public Operation Clone() => CloneBase();

    #endregion
}
