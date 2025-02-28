// Machine Learning Utils
// File name: RnnOperation.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// RNN operation for a neural network.
/// </summary>
public class RnnOperation : Operation
{
    private Matrix? _hiddenState;
    private Matrix? _inputWeights;
    private Matrix? _hiddenWeights;
    private Matrix? _bias;

    public RnnOperation(Matrix inputWeights, Matrix hiddenWeights, Matrix bias)
    {
        _inputWeights = inputWeights;
        _hiddenWeights = hiddenWeights;
        _bias = bias;
        _hiddenState = Matrix.Zeros(1, hiddenWeights.GetDimension(Dimension.Columns));
    }

    protected override Matrix CalcOutput(bool inference)
    {
        Matrix output = Input.MultiplyDot(_inputWeights).Add(_hiddenState.MultiplyDot(_hiddenWeights)).Add(_bias).Tanh();
        _hiddenState = output;
        return output;
    }

    protected override Matrix CalcInputGradient(Matrix outputGradient)
    {
        Matrix tanhGrad = Output.AsOnes().Subtract(Output.MultiplyElementwise(Output));
        Matrix inputGrad = outputGradient.MultiplyElementwise(tanhGrad).MultiplyDot(_inputWeights.Transpose());
        Matrix hiddenGrad = outputGradient.MultiplyElementwise(tanhGrad).MultiplyDot(_hiddenWeights.Transpose());
        return inputGrad.Add(hiddenGrad);
    }

    protected override void EnsureSameShapeForInput(Matrix? input, Matrix inputGradient)
        => EnsureSameShape(input, inputGradient);

    protected override void EnsureSameShapeForOutput(Matrix? output, Matrix outputGradient)
        => EnsureSameShape(output, outputGradient);
}
