// Machine Learning Utils
// File name: LayerBuilder.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed.NeuralNetwork.Layers;

namespace MachineLearning.Typed.NeuralNetwork;

public abstract class LayerBuilder(LayerBuilder? parent)
{
    public LayerBuilder? Parent => parent;

    public Layer Layer { get; protected set; } = null!;
}

public class LayerBuilder<TIn, TOut> : LayerBuilder
    where TIn : notnull
    where TOut : notnull
{
    public LayerBuilder(Layer<TIn, TOut> layer, LayerBuilder? parent = null) : base(parent)
    {
        Layer = layer;
    }

    public LayerBuilder<TOut, TLayerOut> AddLayer<TLayerOut>(Layer<TOut, TLayerOut> layer)
        where TLayerOut : notnull
        => new(layer, this);

    public LayerList<TInputData, TOut> Build<TInputData>()
        where TInputData : notnull
    {
        // Traverse the builder tree to get all layers in the reverse order
        LayerList<TInputData, TOut> layers = [];

        LayerBuilder? builder = this;
        while (builder != null)
        {
            layers.Insert(0, builder.Layer);
            builder = builder.Parent;
        }

        return layers;
    }
}