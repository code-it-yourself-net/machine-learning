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

    internal readonly Matrix Add(float scalar)
    {
        Span<float> result = new(new float[_span.Length]);
        TensorPrimitives.Add(_span, scalar, result);
        return new Matrix(result, _shape);
    }

    internal float GetValue(int row, int col)
    {
        int length = _span.Length;

        Debug.Assert(row * length + col < length, "Index out of range.");
        Debug.Assert(row >= 0 && col >= 0, "Index out of range.");
        
        return _span[row * length + col];
    }
}
