// Machine Learning Utils
// File name: Matrix.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;
using System.Numerics.Tensors;

namespace MachineLearning.Numerics;

public readonly ref struct Matrix
{
    private readonly Span<float> _span;
    private readonly ReadOnlySpan<nint> _shape;

    public unsafe Matrix(float[,] values)
    {
        int rows = values.GetLength(0);
        int cols = values.GetLength(1);

        _shape = new ReadOnlySpan<nint>([rows, cols]);

        int length = rows * cols;

        fixed (float* p = values)
        {
            _span = new(p, length);
        }
    }

    private Matrix(Span<float> span, ReadOnlySpan<nint> shape)
    {
        _span = span;
        _shape = shape;
    }

    public readonly Matrix Add(float scalar)
    {
        Span<float> result = new(new float[_span.Length]);
        TensorPrimitives.Add(_span, scalar, result);
        return new Matrix(result, _shape);
    }

    internal float GetValue(int row, int col)
    {
        Debug.Assert(row >= 0 && row < _shape[0], "Row index is out of range.");
        Debug.Assert(col >= 0 && col < _shape[1], "Column index is out of range.");
        
        return _span[row * (int)_shape[1] + col];

    }

    public readonly Matrix Sigmoid()
    {
        Span<float> result = new(new float[_span.Length]);
        TensorPrimitives.Sigmoid(_span, result);
        return new Matrix(result, _shape);
    }
}
