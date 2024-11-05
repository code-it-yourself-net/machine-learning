// Machine Learning Utils
// File name: Layer.cs
// Code It Yourself with .NET, 2024

/*
namespace MachineLearning.Experimental;

#region Layers

public abstract class Layer
{
    public abstract Type GetOutputType();
    public abstract Type GetInputType();
    internal abstract object Forward(object input);
}

public abstract class Layer<TIn, TOut> : Layer 
    where TIn : notnull
    where TOut : notnull
{
    public abstract TOut Forward(TIn input);

    public abstract TIn Backward(TOut output);

    public override Type GetOutputType() => typeof(TOut);

    public override Type GetInputType() => typeof(TIn);

    internal override object Forward(object input) => Forward((TIn)input);
}

public class DenseLayer : Layer<float[], float[]>
{
    private readonly int _neurons;

    public DenseLayer(int neurons)
    {
        _neurons = neurons;
    }

    public override float[] Backward(float[] output) => throw new NotImplementedException();

    public override float[] Forward(float[] input) => throw new NotImplementedException();
}

public class FlattenLayer : Layer<float[,,,], float[]>
{
    public override float[,,,] Backward(float[] output) => throw new NotImplementedException();

    public override float[] Forward(float[,,,] input) {
        // Flatten the 4D array to 1D array
        float[] result = new float[input.Length];
        int index = 0;
        foreach (float value in input)
        {
            result[index++] = value;
        }
        return result;
    }
}

public class ConvolutionLayer : Layer<float[,,,], float[,,,]>
{
    public override float[,,,] Backward(float[,,,] output) => throw new NotImplementedException();

    public override float[,,,] Forward(float[,,,] input) => throw new NotImplementedException();
}

#endregion Layers

#region NeuralNetwork

public abstract class NeuralNetwork<TInputData, TPrediction> 
    where TInputData : notnull
    where TPrediction : notnull
{
    protected Layers<TInputData, TPrediction> Layers;

    public abstract LayerBuilder<TPrediction> OnAddLayers(LayerBuilder<TInputData> builder);

    public NeuralNetwork()
    {
        Layers =
            OnAddLayers(new LayerBuilder<TInputData>(null))
                .AsLayers<TInputData>();
    }

    public TPrediction Forward(TInputData input)
    {
        TPrediction output = Layers.Forward(input);
        return output;
        //object output = input;
        //foreach (Layer layer in Layers)
        //{
        //    output = layer.Forward(input);
        //}
        //return (TPrediction)output;
    }

}

public class NeuralNetworkConvolution : NeuralNetwork<float[,,,], float[]>
{
    public override LayerBuilder<float[]> OnAddLayers(LayerBuilder<float[,,,]> builder)
    {
        return builder
            .AddLayer(new ConvolutionLayer())
            .AddLayer(new FlattenLayer())
            .AddLayer(new DenseLayer(neurons: 10));
    }

}

#endregion NeuralNetwork



public class Layers<TIn, TOut> : List<Layer>
    where TIn : notnull
    where TOut : notnull
{
    public TOut Forward(TIn input) {
        object output = input;
        foreach (Layer layer in this)
        {
            output = layer.Forward(output);
        }
        return (TOut)output;
    }
}



#region Tests

public class IntToString : Layer<int, string>
{
    public override string Forward(int input) => input.ToString();

    public override int Backward(string output) => int.Parse(output);
}

public class StringToDateTime : Layer<string, DateTime>
{
    public override DateTime Forward(string input) => DateTime.Parse(input);

    public override string Backward(DateTime output) => output.ToString();
}

public class NeuralNetworkIntToDateTime : NeuralNetwork<int, DateTime>
{
    public override LayerBuilder<DateTime> OnAddLayers(LayerBuilder<int> builder)
    {
        return builder
            .AddLayer(new IntToString())
            .AddLayer(new StringToDateTime());
    }
}

#endregion Tests


*/