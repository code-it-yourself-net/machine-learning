// Machine Learning Utils
// File name: NumberWithGradTests.cs
// Code It Yourself with .NET, 2024

using MachineLearning.Typed;

namespace MachineLearning.Tests.Typed;

[TestClass]
public class NumberWithGradTests
{
    [TestMethod]
    public void AddTest()
    {
        // create two NumberWithGrad objects
        NumberWithGrad a = 1;
        NumberWithGrad b = 2;
        // add the two objects
        NumberWithGrad c = a + b;
        // check the result
        Assert.AreEqual(3, c.Number);
    }

    [TestMethod]
    public void MultiplyTest()
    {
        // create two NumberWithGrad objects
        NumberWithGrad a = 2;
        NumberWithGrad b = 3;
        // multiply the two objects
        NumberWithGrad c = a * b;
        // check the result
        Assert.AreEqual(6, c.Number);
    }

    [TestMethod]
    public void GradTest1()
    {
        // Arrange
        NumberWithGrad a = 3;
        NumberWithGrad b = a * 4;
        NumberWithGrad c = b + 5;

        // Act
        c.Backward();

        // Assert
        Assert.AreEqual(4, a.Grad);
        Assert.AreEqual(1, b.Grad);
    }

    [TestMethod]
    public void GradTest2()
    {
        // Arrange
        NumberWithGrad a = 3;
        NumberWithGrad b = a * 4;
        NumberWithGrad c = b + 3;
        NumberWithGrad d = a + 2;
        NumberWithGrad e = c * d;

        // Act
        e.Backward();

        // Assert
        Assert.AreEqual(35, a.Grad);
    }
}
