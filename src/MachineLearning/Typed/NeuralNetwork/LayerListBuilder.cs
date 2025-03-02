﻿// Machine Learning Utils
// File name: LayerBuilder.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed.NeuralNetwork.Layers;

namespace MachineLearning.Typed.NeuralNetwork;

public abstract class LayerListBuilder(LayerListBuilder? parent = null)
{
    public LayerListBuilder? Parent => parent;

    public Layer Layer { get; protected set; } = null!;
}

public class LayerListBuilder<TIn, TOut> : LayerListBuilder
    where TIn : notnull
    where TOut : notnull
{
    internal LayerListBuilder(Layer<TIn, TOut> layer) : base()
    {
        Layer = layer;
    }

    private LayerListBuilder(Layer layer, LayerListBuilder parent) : base(parent)
    {
        Layer = layer;
    }

    public LayerListBuilder<TIn, TNextOut> AddLayer<TNextOut>(Layer<TOut, TNextOut> layer)
        where TNextOut : notnull
        => new(layer, this);

    public LayerList<TIn, TOut> Build()
    {
        // Traverse the builder tree to get all layers in the reverse order
        LayerList<TIn, TOut> layers = [];

        LayerListBuilder? builder = this;
        while (builder != null)
        {
            layers.Insert(0, builder.Layer);
            builder = builder.Parent;
        }

        return layers;
    }
}