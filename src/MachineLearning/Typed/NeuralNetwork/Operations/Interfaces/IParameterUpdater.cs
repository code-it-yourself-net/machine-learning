// Machine Learning Utils
// File name: IParameterUpdater.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed.NeuralNetwork.Layers;
using MachineLearning.Typed.NeuralNetwork.Optimizers;

namespace MachineLearning.Typed.NeuralNetwork.Operations.Interfaces;

internal interface IParameterUpdater
{
    void UpdateParams(Layer? layer, Optimizer optimizer);
}
