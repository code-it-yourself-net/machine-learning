// Machine Learning Utils
// File name: TypedVsUntypedVsFlat.cs
// Code It Yourself with .NET, 2024

using System.Numerics.Tensors;

using BenchmarkDotNet.Attributes;

using MachineLearning;
using MachineLearning.Typed;

namespace MatrixBenchmark;

#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

public class TypedVsUntypedVsFlat
{
    private MatrixOld _matrix1Untyped = null!;
    private MatrixOld _matrix2Untyped = null!;

    private Matrix _matrix1Typed = null!;
    private Matrix _matrix2Typed = null!;

    private float[,] _array1 = null!;
    private float[,] _array2 = null!;

    private readonly Tensor<float> _tensor1 = null!;
    private readonly Tensor<float> _tensor2 = null!;

    private float[] _flattenedArray1 = null!;

    // [Params(100, 1000)]
    [Params(100, 1000)]
    public int N;

    [GlobalSetup]
    public void Setup()
    {
        float[,] matrix1 = new float[790, 89];
        float[,] matrix2 = new float[89, 10];

        // fill in matrix1 and matrix2 with random float numbers
        Random random = new(909);
        for (int i = 0; i < matrix1.GetLength(0); i++)
        {
            for (int j = 0; j < matrix1.GetLength(1); j++)
            {
                matrix1[i, j] = (float)random.NextDouble();
            }
        }

        for (int i = 0; i < matrix2.GetLength(0); i++)
        {
            for (int j = 0; j < matrix2.GetLength(1); j++)
            {
                matrix2[i, j] = (float)random.NextDouble();
            }
        }

        _matrix1Untyped = new((float[,])matrix1.Clone());
        _matrix2Untyped = new((float[,])matrix2.Clone());

        _matrix1Typed = new((float[,])matrix1.Clone());
        _matrix2Typed = new((float[,])matrix2.Clone());

        _array1 = (float[,])matrix1.Clone();
        _array2 = (float[,])matrix2.Clone();

        int rows = matrix1.GetLength(0);
        int cols = matrix1.GetLength(1);
        _flattenedArray1 = new float[rows * cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                _flattenedArray1[i * cols + j] = matrix1[i, j];
            }
        }
    }


    [Benchmark]
    public void UntypedMatrixMultiplication()
    {
        MatrixOld result = _matrix1Untyped.MultiplyDot(_matrix2Untyped);
    }

    [Benchmark]
    public void TypedMatrixMultiplication()
    {
        Matrix result = _matrix1Typed.MultiplyDot(_matrix2Typed);
    }

    [Benchmark]
    public void ArrayMatrixMultiplication()
    {
        float[,] result = _array1.MultiplyDot(_array2);
    }

    //[Benchmark]
    //public void TypedMatrixMultiplicationWithMatrixArray()
    //{
    //    Matrix result = _matrix1Typed.MultiplyDot(_matrix2Typed);
    //}

    [Benchmark]
    public void UntypedSigmoid()
    {
        MatrixOld result = _matrix1Untyped.Sigmoid();
    }

    [Benchmark]
    public void TypedSigmoid()
    {
        Matrix result = _matrix1Typed.Sigmoid();
    }

    [Benchmark]
    public void ArraySigmoid()
    {
        float[,] result = _array1.Sigmoid();
    }

    [Benchmark]
    public void TensorPrimitivesSigmoid()
    {
        Span<float> dest = new(new float[_flattenedArray1.Length]);
        TensorPrimitives.Sigmoid(new ReadOnlySpan<float>(_flattenedArray1), dest);
    }

    //[Benchmark]
    //public void Softmax()
    //{
    //    TypedMatrix sm = _matrix1Typed.Softmax();
    //}

    //[Benchmark]
    //public void SoftmaxWithCache()
    //{
    //    TypedMatrix sm = _matrix1Typed.SoftmaxWithCache();
    //}

    /*
    [Benchmark]
    public void MaxLoopTyped()
    {
        float max = _matrix1Typed.MaxLoopTyped();
    }
    */
}

#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
