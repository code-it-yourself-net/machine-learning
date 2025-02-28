// Machine Learning Utils
// File name: NeuralNetworkTests.cs
// Code It Yourself with .NET, 2024

using MachineLearning.NeuralNetwork.Layers;
using MachineLearning.NeuralNetwork.Losses;
using MachineLearning.NeuralNetwork.Operations;
using MachineLearning.NeuralNetwork.ParamInitializers;

namespace MachineLearning.NeuralNetwork.Tests;

[TestClass]
public class NeuralNetworkTests
{
    [TestMethod]
    public void CloneTest()
    {
        // Arrange
        SeededRandom random = new(12345);
        int xColumns = 89, yColumns = 10, rows = 2;
        NeuralNetwork neuralNetwork = new(
            [
                new DenseLayer(xColumns, new Tanh(), new RandomInitializer(random)),
                new DenseLayer(yColumns, new Linear(), new RandomInitializer(random))
            ],
            new SoftmaxCrossEntropyLoss()
        );

        neuralNetwork.TrainBatch(Matrix.Ones(rows, xColumns), Matrix.Ones(rows, yColumns));

        // Act
        NeuralNetwork clonedNeuralNetwork = neuralNetwork.Clone();

        // Assert
        Assert.AreEqual(neuralNetwork.LossFunction.GetType(), clonedNeuralNetwork.LossFunction.GetType());
        Assert.IsTrue(neuralNetwork.LossFunction.Prediction.HasSameValues(clonedNeuralNetwork.LossFunction.Prediction));
        Assert.IsTrue(neuralNetwork.LossFunction.Target.HasSameValues(clonedNeuralNetwork.LossFunction.Target));
        Assert.AreEqual(neuralNetwork.LastLoss, clonedNeuralNetwork.LastLoss);
        Assert.AreEqual(neuralNetwork.GetAllParams().Length, clonedNeuralNetwork.GetAllParams().Length);
        Assert.IsTrue(neuralNetwork.GetAllParams()[0].HasSameValues(clonedNeuralNetwork.GetAllParams()[0]));
        Assert.AreEqual(neuralNetwork.GetAllParamGradients().Length, clonedNeuralNetwork.GetAllParamGradients().Length);
        Assert.IsTrue(neuralNetwork.GetAllParamGradients()[0].HasSameValues(clonedNeuralNetwork.GetAllParamGradients()[0]));
        Assert.AreEqual(neuralNetwork.HasCheckpoint(), clonedNeuralNetwork.HasCheckpoint());
        Assert.AreEqual(neuralNetwork.ParameterCount, clonedNeuralNetwork.ParameterCount);
    }

    [TestMethod]
    public void RnnLayerForwardBackwardTest()
    {
        // Arrange
        int inputSize = 5;
        int hiddenSize = 5;
        int batchSize = 3;
        int timeSteps = 4;

        Matrix input = Matrix.Random(batchSize, inputSize);
        Matrix outputGradient = Matrix.Random(batchSize, hiddenSize);

        RnnLayer rnnLayer = new();

        // Act
        Matrix output = rnnLayer.Forward(input, false);
        Matrix inputGradient = rnnLayer.Backward(outputGradient);

        // Assert
        Assert.AreEqual(output.GetDimension(Dimension.Rows), batchSize);
        Assert.AreEqual(output.GetDimension(Dimension.Columns), hiddenSize);
        Assert.AreEqual(inputGradient.GetDimension(Dimension.Rows), batchSize);
        Assert.AreEqual(inputGradient.GetDimension(Dimension.Columns), inputSize);
    }

    [TestMethod]
    public void RnnOperationForwardBackwardTest()
    {
        // Arrange
        int inputSize = 5;
        int hiddenSize = 5;
        int batchSize = 3;

        Matrix input = Matrix.Random(batchSize, inputSize);
        Matrix outputGradient = Matrix.Random(batchSize, hiddenSize);

        Matrix inputWeights = Matrix.Random(inputSize, hiddenSize);
        Matrix hiddenWeights = Matrix.Random(hiddenSize, hiddenSize);
        Matrix bias = Matrix.Zeros(1, hiddenSize);

        RnnOperation rnnOperation = new(inputWeights, hiddenWeights, bias);

        // Act
        Matrix output = rnnOperation.Forward(input, false);
        Matrix inputGradient = rnnOperation.Backward(outputGradient);

        // Assert
        Assert.AreEqual(output.GetDimension(Dimension.Rows), batchSize);
        Assert.AreEqual(output.GetDimension(Dimension.Columns), hiddenSize);
        Assert.AreEqual(inputGradient.GetDimension(Dimension.Rows), batchSize);
        Assert.AreEqual(inputGradient.GetDimension(Dimension.Columns), inputSize);
    }
}
