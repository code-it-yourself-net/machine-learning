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

    private const float scalar = 2.94f;

    // private MachineLearning.Numerics.Matrix _matrix1 = null!;

    //private readonly Tensor<float> _tensor1 = null!;
    //private readonly Tensor<float> _tensor2 = null!;

    //private float[] _flattenedArray1 = null!;

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

        /*
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

        float[] vectorData = { 1.0f, 2.0f, 3.0f, 4.0f };

        // Convert tensor1 to ReadOnlySpan<float>
        
        Tensor<float> tensor1 = Tensor.Create(vectorData, new ReadOnlySpan<nint>([vectorData.Length]));
        //tensor1.AsReadOnlyTensorSpan()
        //Tensor<float> vector = new Tensor<float>(vectorData, new[] { vectorData.Length }); // Shape: [4]
        */
    }

    [Benchmark]
    public void UntypedAddScalar()
    {
        _ = _matrix1Untyped.Add(scalar);
    }

    [Benchmark]
    public void TypedAddScalar()
    {
        _ = _matrix1Typed.Add(scalar);
    }

    [Benchmark]
    public void ArrayAddScalar()
    {
        _ = _array1.Add(scalar);
    }

    [Benchmark]
    public void NumericMatrixAddScalar() 
    {
        MachineLearning.Numerics.Matrix matrix = new(_array1);
        _ = matrix.Add(scalar);
    }


    /*
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
    */
    //[Benchmark]
    //public void TypedMatrixMultiplicationWithMatrixArray()
    //{
    //    Matrix result = _matrix1Typed.MultiplyDot(_matrix2Typed);
    //}

    
    [Benchmark]
    public void UntypedSigmoid()
    {
        _ = _matrix1Untyped.Sigmoid();
    }

    [Benchmark]
    public void TypedSigmoid()
    {
        _ = _matrix1Typed.Sigmoid();
    }

    [Benchmark]
    public void ArraySigmoid()
    {
        _ = _array1.Sigmoid();
    }
    
    [Benchmark]
    public void TensorPrimitivesSigmoid()
    {
        MachineLearning.Numerics.Matrix matrix = new(_array1);
        _ = matrix.Sigmoid();
    }

    /*
    [Benchmark]
    public void TensorPrimitivesSigmoid2()
    {
        int rows = _array1.GetLength(0);
        int cols = _array1.GetLength(1);
        float[] flattenedArray1 = new float[rows * cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                flattenedArray1[i * cols + j] = _array1[i, j];
            }
        }

        Span<float> dest = new(new float[flattenedArray1.Length]);
        TensorPrimitives.Sigmoid(new ReadOnlySpan<float>(flattenedArray1), dest);

        // unflatten the result
        float[,] result = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = dest[i * cols + j];
            }
        }
    }*/

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
