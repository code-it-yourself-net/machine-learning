// Machine Learning Utils
// File name: OperationBuilder.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed.NeuralNetwork.Operations;

namespace MachineLearning.Typed.NeuralNetwork.Layers;

/*
 *  OperationBuilder and OperationBuilder<TIn, TOut> are in the Layers namespace, because they are used to build layers.
 */

public abstract class OperationListBuilder
{
    private readonly OperationListBuilder? _parent;

    public OperationListBuilder(OperationListBuilder? parent)
    {
        _parent = parent;
    }

    public OperationListBuilder? Parent => _parent;

    public Operation Operation { get; protected set; }
}

//public class ForInput<TLayerIn>
//    where TLayerIn : notnull
//{
//    public OperationListBuilder<TLayerIn, TOpOut> AddOperation<TOpOut>(Operation<TLayerIn, TOpOut> operation)
//        where TOpOut : notnull
//    {
//        OperationListBuilder<TLayerIn, TOpOut> builder = new(operation);
//        return builder;
//    }
//}

public class OperationListBuilder<TIn, TOut> : OperationListBuilder
    where TIn : notnull
    where TOut : notnull
{
    public OperationListBuilder(Operation<TIn, TOut> operation, OperationListBuilder? parent = null) : base(parent)
    {
        Operation = operation;
    }

    public OperationListBuilder<TOut, TOpOut> AddOperation<TOpOut>(Operation<TOut, TOpOut> operation)
        where TOpOut : notnull
    {
        OperationListBuilder<TOut, TOpOut> builder = new(operation, this);
        return builder;
    }

    public OperationList<TLayerIn, TOut> Build<TLayerIn>()
        where TLayerIn : notnull
    {
        // Traverse the builder chain backwards to get all the operations in the reverse order
        OperationList<TLayerIn, TOut> operations = [];

        OperationListBuilder? builder = this;
        while (builder != null)
        {
            operations.Insert(0, builder.Operation);
            builder = builder.Parent;
        }

        return operations;
    }
}
