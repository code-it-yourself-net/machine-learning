﻿// Machine Learning Utils
// File name: Layer.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

using MachineLearning.Typed.NeuralNetwork.Optimizers;

namespace MachineLearning.Typed.NeuralNetwork.Layers;

public abstract class Layer
{
    public abstract Type GetOutputType();
    public abstract Type GetInputType();
    public abstract object Forward(object input, bool inference);
    public abstract object Backward(object outputGradient);
    public abstract void UpdateParams(Optimizer optimizer);
}

public abstract class Layer<TIn, TOut> : Layer
    where TIn : notnull
    where TOut : notnull
{
    private bool _first = true;
    private TOut? _output;
    private TIn? _input;

    private OperationList<TIn, TOut>? _operations;

    protected TIn? Input => _input;

    /// <summary>
    /// Passes input forward through a series of operations.
    /// </summary>
    /// <param name="input">Input matrix.</param>
    /// <returns>Output matrix.</returns>
    public TOut Forward(TIn input, bool inference)
    {
        // We store the pointer to the input array so we can check the shape of the input gradient in the backward pass..
        _input = input;
        if (_first)
        {
            SetupLayer(input);
            _first = false;
        }

        Debug.Assert(_operations != null, "Operations were not set up.");

        // As above, we store the pointer to the output array so we can check the shape of the output gradient in the backward pass.
        _output = _operations.Forward(input, inference);

        return _output;
    }

    /// <summary>
    /// Passes <paramref name="outputGradient"/> backward through a series of operations.
    /// </summary>
    /// <remarks>
    /// Checks appropriate shapes. 
    /// </remarks>
    public TIn Backward(TOut outputGradient)
    {
        EnsureSameShapeForOutput(_output, outputGradient);

        Debug.Assert(_operations != null, "Operations were not set up.");

        TIn inputGradient = _operations.Backward(outputGradient);

        //_paramGradients = Operations
        //    .OfType<ParamOperation>()
        //    .Select(po => po.ParamGradient)
        //    .ToList();

        EnsureSameShapeForInput(_input, inputGradient);

        return inputGradient;
    }

    public override void UpdateParams(Optimizer optimizer)
    {
        Debug.Assert(_operations != null, "Operations were not set up.");

        _operations.UpdateParams(this, optimizer);
    }

    public abstract OperationBuilder<TOut> OnAddOperations(OperationBuilder<TIn> builder);

    protected virtual void SetupLayer(TIn input)
    {
        _operations =
            OnAddOperations(new OperationBuilder<TIn>(null))
                .AsOperationList<TIn>();
    }

    public override Type GetOutputType() => typeof(TOut);

    public override Type GetInputType() => typeof(TIn);

    public override object Forward(object input, bool inference) => Forward((TIn)input, inference);

    public override object Backward(object outputGradient) => Backward((TOut)outputGradient);

    [Conditional("DEBUG")]
    protected abstract void EnsureSameShapeForInput(TIn? input, TIn? inputGradient);

    [Conditional("DEBUG")]
    protected abstract void EnsureSameShapeForOutput(TOut? output, TOut? outputGradient);
}