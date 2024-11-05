// Machine Learning Utils
// File name: DataSource.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.Typed.NeuralNetwork.DataSources;

public abstract class DataSource<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    public abstract (TInputData xTrain, TPrediction yTrain, TInputData? xTest, TPrediction? yTest) GetData();
}
