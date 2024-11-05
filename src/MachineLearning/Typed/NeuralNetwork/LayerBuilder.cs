// Machine Learning Utils
// File name: LayerBuilder.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed.NeuralNetwork.Layers;

namespace MachineLearning.Typed.NeuralNetwork;

public class LayerBuilder
{
    private readonly LayerBuilder? _parent;

    public LayerBuilder(LayerBuilder? parent)
    {
        _parent = parent;
    }

    public LayerBuilder? Parent => _parent;

    public Layer? Layer { get; protected set; }
}

public class LayerBuilder<TIn> : LayerBuilder
    where TIn : notnull
{
    public LayerBuilder(LayerBuilder? parent) : base(parent)
    {
    }

    public LayerBuilder<TOut> AddLayer<TOut>(Layer<TIn, TOut> layer)
        where TOut : notnull
    {
        Layer = layer;
        return new LayerBuilder<TOut>(this);
    }

    public LayerList<TInputData, TIn> AsLayerList<TInputData>()
        where TInputData : notnull
    {
        // Traverse the builder tree to get all layers in the reverse order
        LayerList<TInputData, TIn> layers = [];

        LayerBuilder? builder = this;
        while (builder != null)
        {
            if (builder.Layer != null)
            {
                layers.Insert(0, builder.Layer);
            }
            builder = builder.Parent;
        }
        return layers;
    }
}