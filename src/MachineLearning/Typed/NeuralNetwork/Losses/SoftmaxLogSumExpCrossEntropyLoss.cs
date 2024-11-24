﻿// Machine Learning Utils
// File name: SoftmaxCrossEntropyLoss.cs
// Code It Yourself with .NET, 2024

namespace MachineLearning.Typed.NeuralNetwork.Losses;

public class SoftmaxLogSumExpCrossEntropyLoss(float eps = 1e-7f) : Loss2D
{
    protected override float CalculateLoss()
    {
        // Calculate the probabilities for the whole batch.
        // TODO: softmaxPrediction can be stored (before ClipInPlace) and reused in CalculateLossGradient.
        float[,] softmaxPrediction = Prediction.SoftmaxLogSumExp();

        // Clip the probabilities to avoid log(0).
        softmaxPrediction.ClipInPlace(eps, 1 - eps);

        float[,] negativeTarget = Target.Multiply(-1f);
        float[,] softmaxCrossEntropyLoss = negativeTarget.MultiplyElementwise(softmaxPrediction.Log())
            .Subtract(
                negativeTarget.Add(1f).MultiplyElementwise(softmaxPrediction.Multiply(-1f).Add(1f).Log())
            );
        int batchSize = Prediction.GetLength((int)Dimension.Rows);
        return softmaxCrossEntropyLoss.Sum() / batchSize;
    }

    protected override float[,] CalculateLossGradient()
    {
        float[,] softmaxPrediction = Prediction.SoftmaxLogSumExp();
        int batchSize = Prediction.GetLength((int)Dimension.Rows);
        return softmaxPrediction.Subtract(Target).Divide(batchSize);
    }
}
