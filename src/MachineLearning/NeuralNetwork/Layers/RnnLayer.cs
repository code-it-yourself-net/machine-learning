// Machine Learning Utils
// File name: RnnLayer.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.Operations;

namespace MachineLearning.NeuralNetwork.Layers;

/// <summary>
/// A "layer" of neurons in a neural network.
/// </summary>
public class RnnLayer : Layer
{
    private Matrix? _hiddenState;
    private Matrix? _inputWeights;
    private Matrix? _hiddenWeights;
    private Matrix? _bias;

    protected override void SetupLayer(Matrix input)
    {
        int inputSize = input.GetDimension(Dimension.Columns);
        int hiddenSize = inputSize; // Assuming hidden size is the same as input size for simplicity

        _inputWeights = Matrix.Random(inputSize, hiddenSize);
        _hiddenWeights = Matrix.Random(hiddenSize, hiddenSize);
        _bias = Matrix.Zeros(1, hiddenSize);
        _hiddenState = Matrix.Zeros(1, hiddenSize);

        Params.Add(_inputWeights);
        Params.Add(_hiddenWeights);
        Params.Add(_bias);
    }

    public override Matrix Forward(Matrix input, bool inference)
    {
        if (_first)
        {
            SetupLayer(input);
            _first = false;
        }

        Matrix output = input.MultiplyDot(_inputWeights).Add(_hiddenState.MultiplyDot(_hiddenWeights)).Add(_bias).Tanh();
        _hiddenState = output;

        _output = output;
        return _output;
    }

    public override Matrix Backward(Matrix outputGradient)
    {
        EnsureSameShape(_output, outputGradient);

        Matrix tanhGrad = _output.AsOnes().Subtract(_output.MultiplyElementwise(_output));
        Matrix inputGrad = outputGradient.MultiplyElementwise(tanhGrad).MultiplyDot(_inputWeights.Transpose());
        Matrix hiddenGrad = outputGradient.MultiplyElementwise(tanhGrad).MultiplyDot(_hiddenWeights.Transpose());

        _paramGradients = new List<Matrix>
        {
            _inputWeights.Transpose().MultiplyDot(outputGradient.MultiplyElementwise(tanhGrad)),
            _hiddenWeights.Transpose().MultiplyDot(outputGradient.MultiplyElementwise(tanhGrad)),
            outputGradient.MultiplyElementwise(tanhGrad).SumBy(Dimension.Rows)
        };

        return inputGrad.Add(hiddenGrad);
    }
}
