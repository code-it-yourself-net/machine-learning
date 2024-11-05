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
}
