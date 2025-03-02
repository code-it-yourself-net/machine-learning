﻿// Machine Learning Utils
// File name: Layer.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

using MachineLearning.Typed.NeuralNetwork.Operations;
using MachineLearning.Typed.NeuralNetwork.Optimizers;

namespace MachineLearning.Typed.NeuralNetwork.Layers;

public abstract class Layer
{
    public abstract Type GetOutputType();
    public abstract Type GetInputType();
    public abstract object Forward(object input, bool inference);
    public abstract object Backward(object outputGradient);
    public abstract void UpdateParams(Optimizer optimizer);
    public abstract int GetParamCount();
}

public abstract class Layer<TIn, TOut> : Layer
    where TIn : notnull
    where TOut : notnull
{
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
        bool firstPass = _input is null;

        // We store the pointer to the input array so we can check the shape of the input gradient in the backward pass.
        _input = input;
        if (firstPass)
        {
            // First pass, set up the layer.
            SetupLayer(input);
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

    public abstract OperationListBuilder<TIn, TOut> CreateOperationsBuilder();

    protected virtual void SetupLayer(TIn input)
    {
        // Build the operations list.
        _operations = CreateOperationsBuilder().Build<TIn>();
    }

    protected static OperationListBuilder<TIn, TOpOut> AddOperation<TOpOut>(Operation<TIn, TOpOut> operation)
        where TOpOut : notnull
    {
        OperationListBuilder<TIn, TOpOut> builder = new(operation);
        return builder;
    }

    public override Type GetOutputType() => typeof(TOut);

    public override Type GetInputType() => typeof(TIn);

    public override object Forward(object input, bool inference) => Forward((TIn)input, inference);

    public override object Backward(object outputGradient) => Backward((TOut)outputGradient);

    [Conditional("DEBUG")]
    protected abstract void EnsureSameShapeForInput(TIn? input, TIn? inputGradient);

    [Conditional("DEBUG")]
    protected abstract void EnsureSameShapeForOutput(TOut? output, TOut? outputGradient);

    public override int GetParamCount()
    {
        Debug.Assert(_operations != null, "Operations were not set up.");

        return _operations.GetParamCount();
    }

}
