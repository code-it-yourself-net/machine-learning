// Machine Learning Utils
// File name: OperationBuilder.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed.NeuralNetwork.Operations;

namespace MachineLearning.Typed.NeuralNetwork.Layers;

/*
 *  OperationBuilder and OperationBuilder<TIn> are in the Layers namespace, because they are used to build layers.
 */

public class OperationListBuilder
{
    private readonly OperationListBuilder? _parent;

    public OperationListBuilder(OperationListBuilder? parent)
    {
        _parent = parent;
    }

    public OperationListBuilder? Parent => _parent;

    public Operation? Operation { get; protected set; }
}

public class OperationListBuilder<TIn> : OperationListBuilder
    where TIn : notnull
{
    public OperationListBuilder(OperationListBuilder? parent) : base(parent)
    {
    }

    public OperationListBuilder<TOut> AddOperation<TOut>(Operation<TIn, TOut> operation)
        where TOut : notnull
    {
        Operation = operation;
        return new OperationListBuilder<TOut>(this);
    }

    public OperationList<TInputData, TIn> Build<TInputData>()
        where TInputData : notnull
    {
        // Traverse the builder tree to get all the operations in the reverse order
        OperationList<TInputData, TIn> operations = [];

        OperationListBuilder? builder = this;
        while (builder != null)
        {
            if (builder.Operation != null)
            {
                operations.Insert(0, builder.Operation);
            }
            builder = builder.Parent;
        }
        return operations;
    }
}
