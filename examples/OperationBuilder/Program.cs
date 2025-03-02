// Machine Learning Utils
// File name: Program.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed.NeuralNetwork.Layers;
using MachineLearning.Typed.NeuralNetwork.Operations;

class IntToStringOperation : Operation<int, string>
{
    protected override int CalcInputGradient(string outputGradient) => throw new NotImplementedException();
    protected override string CalcOutput(bool inference) => throw new NotImplementedException();
    protected override void EnsureSameShapeForInput(int input, int inputGradient) => throw new NotImplementedException();
    protected override void EnsureSameShapeForOutput(string? output, string outputGradient) => throw new NotImplementedException();
}

class IntToStringLayer : Layer<int, string>
{
    public override OperationListBuilder<int, string> CreateOperationsBuilder()
    {
        return AddOperation(new IntToStringOperation());
    }

    protected override void EnsureSameShapeForInput(int input, int inputGradient) => throw new NotImplementedException();
    protected override void EnsureSameShapeForOutput(string? output, string? outputGradient) => throw new NotImplementedException();
}


internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
    }
}