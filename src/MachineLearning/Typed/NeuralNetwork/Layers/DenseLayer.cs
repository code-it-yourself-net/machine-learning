// Machine Learning Utils
// File name: DenseLayer.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;

using MachineLearning.Typed.NeuralNetwork.Operations;
using MachineLearning.Typed.NeuralNetwork.ParamInitializers;

using static MachineLearning.Typed.ArrayUtils;

namespace MachineLearning.Typed.NeuralNetwork.Layers;

public class DenseLayer : Layer<float[,], float[,]>
{
    private readonly int _neurons;
    private readonly Operation2D _activationFunction;
    private readonly ParamInitializer _paramInitializer;
    private readonly Dropout? _dropout;

    public DenseLayer(int neurons, Operation2D activationFunction, ParamInitializer paramInitializer, Dropout? dropout = null)
    {
        _neurons = neurons;
        _activationFunction = activationFunction;
        _paramInitializer = paramInitializer;
        _dropout = dropout;
    }

    public override OperationBuilder<float[,]> OnAddOperations(OperationBuilder<float[,]> builder)
    {
        Debug.Assert(Input != null);

        float[,] weights = _paramInitializer.InitWeights(Input.GetLength((int)Dimension.Columns), _neurons);
        float[] biases = _paramInitializer.InitBiases(_neurons);

        OperationBuilder<float[,]> res = builder
            .AddOperation(new WeightMultiply(weights))
            .AddOperation(new BiasAdd(biases))
            .AddOperation(_activationFunction);

        if (_dropout != null)
            res = res.AddOperation(_dropout);

        return res;
    }

    protected override void EnsureSameShapeForInput(float[,]? input, float[,]? inputGradient)
        => EnsureSameShape(input, inputGradient);

    protected override void EnsureSameShapeForOutput(float[,]? output, float[,]? outputGradient)
        => EnsureSameShape(output, outputGradient);

    public override string ToString() => $"DenseLayer (neurons={_neurons}, activation={_activationFunction}, paramInitializer={_paramInitializer}, dropout={_dropout})";
}
