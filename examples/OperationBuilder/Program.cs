// Machine Learning Utils
// File name: Program.cs
// Code It Yourself with .NET, 2024

using MachineLearning;
using MachineLearning.Typed.NeuralNetwork;
using MachineLearning.Typed.NeuralNetwork.Layers;
using MachineLearning.Typed.NeuralNetwork.Losses;
using MachineLearning.Typed.NeuralNetwork.Operations;

class IntToStringOperation : Operation<int, string>
{
    protected override int CalcInputGradient(string outputGradient) => throw new NotImplementedException();
    protected override string CalcOutput(bool inference) => throw new NotImplementedException();
    protected override void EnsureSameShapeForInput(int input, int inputGradient) => throw new NotImplementedException();
    protected override void EnsureSameShapeForOutput(string? output, string outputGradient) => throw new NotImplementedException();
}

class StringToStringOperation : Operation<string, string>
{
    protected override string CalcInputGradient(string outputGradient) => throw new NotImplementedException();
    protected override string CalcOutput(bool inference) => throw new NotImplementedException();
    protected override void EnsureSameShapeForInput(string? input, string inputGradient) => throw new NotImplementedException();
    protected override void EnsureSameShapeForOutput(string? output, string outputGradient) => throw new NotImplementedException();
}

class IntToStringLayer : Layer<int, string>
{
    public override OperationListBuilder<int, string> CreateOperationList()
    {
        return AddOperation(new IntToStringOperation())
            .AddOperation(new StringToStringOperation());
    }

    protected override void EnsureSameShapeForInput(int input, int inputGradient) => throw new NotImplementedException();
    protected override void EnsureSameShapeForOutput(string? output, string? outputGradient) => throw new NotImplementedException();
}

class StringToStringLayer : Layer<string, string>
{
    public override OperationListBuilder<string, string> CreateOperationList()
    {
        return AddOperation(new StringToStringOperation())
            .AddOperation(new StringToStringOperation());
    }
    protected override void EnsureSameShapeForInput(string input, string inputGradient) => throw new NotImplementedException();

    protected override void EnsureSameShapeForOutput(string? output, string? outputGradient) => throw new NotImplementedException();
}

class IntToStringNeuralNetword : NeuralNetwork<int, string>
{
    public IntToStringNeuralNetword(Loss<string> lossFunction, SeededRandom? random) : base(lossFunction, random)
    {
    }

    protected override LayerBuilder<int, string> CreateLayerBuilder()
    {
        return AddLayer(new IntToStringLayer())
            .AddLayer(new StringToStringLayer());
    }
}


internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
    }
}