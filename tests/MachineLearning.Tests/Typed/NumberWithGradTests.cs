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
    public void GradTest()
    {
        NumberWithGrad a = 3;
        NumberWithGrad b = a * 4;
        NumberWithGrad c = b + 5;
        c.Backward();

        Assert.AreEqual(4, a.Grad);
        Assert.AreEqual(1, b.Grad);
    }
}
