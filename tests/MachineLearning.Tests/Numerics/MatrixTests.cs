// Machine Learning Utils
// File name: MatrixTests.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.Numerics.Tests;

[TestClass]
public class MatrixTests
{
    [TestMethod]
    public void AddTest()
    {
        Matrix matrix = new(new float[,] { { 1, 2 }, { 3, 4 } });
        Matrix result = matrix.Add(2);
        Assert.AreEqual(3f, result.GetValue(0, 0));
        Assert.AreEqual(6f, result.GetValue(1, 1));
    }
}
