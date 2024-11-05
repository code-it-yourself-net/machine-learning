// Machine Learning Utils
// File name: SimpleDataSource.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.Typed.NeuralNetwork.DataSources;

public class SimpleDataSource<TInputData, TPrediction>(TInputData xTrain, TPrediction yTrain, TInputData? xTest, TPrediction? yTest) : DataSource<TInputData, TPrediction>
    where TInputData : notnull
    where TPrediction : notnull
{
    public override (TInputData xTrain, TPrediction yTrain, TInputData? xTest, TPrediction? yTest) GetData() => (xTrain, yTrain, xTest, yTest);
}
