// Machine Learning Utils
// File name: BiasAdd.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.NeuralNetwork.Operations;

/// <summary>
/// Computes bias addition.
/// </summary>
/// <param name="bias">Bias matrix.</param>
public class BiasAdd(Matrix bias) : ParamOperation(ValidateBiasMatrix(bias))
{
    protected override Matrix CalcOutput(bool inference) => Input.AddRow(Param);

    protected override Matrix CalcInputGradient(Matrix outputGradient) 
        => Matrix.Ones(Input).MultiplyElementwise(outputGradient);

    protected override Matrix CalcParamGradient(Matrix outputGradient)
    {
        Matrix paramGrad = Matrix.Ones(Param).MultiplyElementwise(outputGradient);
        // return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
        // return paramGrad.SumBy(Dimension.Rows); // Reshape - ?
        return paramGrad.AvgBy(Dimension.Rows);
        // return Matrix.Zeros(1, paramGrad.GetDimension(Dimension.Columns)); // .AddRow(paramGrad.SumBy(Dimension.Rows));
    }

    private static Matrix ValidateBiasMatrix(Matrix bias)
    {
#if DEBUG
        if (bias.GetDimension(Dimension.Rows) != 1)
            throw new ArgumentException("Bias matrix must have one row.");
#endif
        return bias;
    }
}
