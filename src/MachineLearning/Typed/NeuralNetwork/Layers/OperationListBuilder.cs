// Machine Learning Utils
// File name: OperationListBuilder.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed.NeuralNetwork.Operations;

namespace MachineLearning.Typed.NeuralNetwork.Layers;

/*
 *  OperationBuilder and OperationBuilder<TIn, TOut> are in the Layers namespace, because they are used to build layers.
 */

public abstract class OperationListBuilder(OperationListBuilder? parent)
{
    public OperationListBuilder? Parent => parent;

    public Operation Operation { get; protected set; } = null!;
}

public class OperationListBuilder<TLayerIn, TLastOut> : OperationListBuilder
    where TLayerIn : notnull
    where TLastOut : notnull
{
    internal OperationListBuilder(Operation<TLayerIn, TLastOut> operation): base(null)
    {
        Operation = operation;
    }

    private OperationListBuilder(Operation operation, OperationListBuilder parent) : base(parent)
    {
        Operation = operation;
    }

    public OperationListBuilder<TLayerIn, TOperationOut> AddOperation<TOperationOut>(Operation<TLastOut, TOperationOut> operation)
        where TOperationOut : notnull
        => new OperationListBuilder<TLayerIn, TOperationOut>(operation, this);

    public OperationList<TLayerIn, TLastOut> Build()
    {
        // Traverse the builder chain backwards to get all the operations in the reverse order
        OperationList<TLayerIn, TLastOut> operations = [];

        OperationListBuilder? builder = this;
        while (builder != null)
        {
            operations.Insert(0, builder.Operation);
            builder = builder.Parent;
        }

        return operations;
    }
}
