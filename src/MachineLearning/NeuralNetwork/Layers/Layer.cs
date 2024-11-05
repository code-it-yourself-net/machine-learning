// Machine Learning Utils
// File name: Layer.cs
// Code It Yourself with .NET, 2024

// This class is derived from the content originally published in the book Deep Learning from Scratch: Building with
// Python from First Principles by Seth Weidman. Some comments here are copied/modified from the original text.

using MachineLearning.NeuralNetwork.Exceptions;
using MachineLearning.NeuralNetwork.Operations;

using static MachineLearning.MatrixUtils;

namespace MachineLearning.NeuralNetwork.Layers;

/// <summary>
/// A "layer" of neurons in a neural network.
/// </summary>
public abstract class Layer()
{
    private bool _first = true;

    /// <summary>
    /// This field is used during the backward pass.
    /// </summary>
    private Matrix? _output;

    /// <summary>
    /// This field is used in <see cref="Optimizers.Optimizer.Step(NeuralNetwork)"/>.
    /// </summary>
    private List<Matrix>? _paramGradients;

    /// <summary>
    /// The parameters (weights & biases) of the layer.
    /// </summary>
    internal List<Matrix> Params { get; private set; } = [];

    protected List<Operation> Operations { get; private set; } = [];

    internal List<Matrix> ParamGradients => _paramGradients ?? throw new NotYetCalculatedException();

    protected abstract void SetupLayer(Matrix input);

    /// <summary>
    /// Passes input forward through a series of operations.
    /// </summary>
    /// <param name="input">Input matrix.</param>
    /// <returns>Output matrix.</returns>
    public Matrix Forward(Matrix input, bool inference)
    {
        if (_first)
        {
            SetupLayer(input);
            _first = false;
        }

        foreach (Operation operation in Operations)
        {
            input = operation.Forward(input, inference);
        }

        _output = input;
        return _output;
    }

    /// <summary>
    /// Passes <paramref name="outputGradient"/> backward through a series of operations.
    /// </summary>
    /// <remarks>
    /// Checks appropriate shapes. 
    /// </remarks>
    public Matrix Backward(Matrix outputGradient)
    {
        EnsureSameShape(_output, outputGradient);

        foreach (Operation operation in Operations.Reverse<Operation>())
        {
            outputGradient = operation.Backward(outputGradient);
        }

        _paramGradients = Operations
            .OfType<ParamOperation>()
            .Select(po => po.ParamGradient)
            .ToList();

        return outputGradient;
    }

    #region Clone

    protected virtual Layer CloneBase()
    {
        Layer clone = (Layer)MemberwiseClone();
        clone._output = _output?.Clone();
        clone._paramGradients = _paramGradients?.Select(p => p.Clone()).ToList();
        clone.Params = Params.Select(p => p.Clone()).ToList();
        clone.Operations = Operations.Select(o => o.Clone()).ToList();
        return clone;
    }

    public Layer Clone() => CloneBase();

    #endregion
}
