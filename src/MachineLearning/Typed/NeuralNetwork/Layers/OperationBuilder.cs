// Machine Learning Utils
// File name: OperationBuilder.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed.NeuralNetwork.Operations;

namespace MachineLearning.Typed.NeuralNetwork.Layers;

/*
 *  OperationBuilder and OperationBuilder<TIn> are in the Layers namespace, because they are used to build layers.
 */

public class OperationBuilder
{
    private readonly OperationBuilder? _parent;

    public OperationBuilder(OperationBuilder? parent)
    {
        _parent = parent;
    }

    public OperationBuilder? Parent => _parent;

    public Operation? Operation { get; protected set; }
}

public class OperationBuilder<TIn> : OperationBuilder
    where TIn : notnull
{
    public OperationBuilder(OperationBuilder? parent) : base(parent)
    {
    }

    public OperationBuilder<TOut> AddOperation<TOut>(Operation<TIn, TOut> operation)
        where TOut : notnull
    {
        Operation = operation;
        return new OperationBuilder<TOut>(this);
    }

    public OperationList<TInputData, TIn> AsOperationList<TInputData>()
        where TInputData : notnull
    {
        // Traverse the builder tree to get all the operations in the reverse order
        OperationList<TInputData, TIn> operations = [];

        OperationBuilder? builder = this;
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
