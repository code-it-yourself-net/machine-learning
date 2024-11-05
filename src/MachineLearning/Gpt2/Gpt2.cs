// Machine Learning Utils
// File name: Gpt2.cs
// Code It Yourself with .NET, 2024

// Port from https://github.com/romanoza/llm.c/blob/master/train_gpt2.c

using System.Diagnostics;

namespace MachineLearning.Gpt2;

public class Gpt2
{
    /* 
     * All the individual layers' forward and backward passes
     * B = batch_size, T = sequence_length, C = channels, V = vocab_size
     */

    /// <param name="output">output is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position</param>
    /// <param name="input">input is (B,T) of integers, holding the token ids at each (b,t) position</param>
    /// <param name="wte">wte is (V,C) of token embeddings, short for "weight token embeddings"</param>
    /// <param name="wpe">wpe is (maxT,C) of position embeddings, short for "weight positional embedding"</param>
    /// <param name="B">batch_size</param>
    /// <param name="T">sequence_length</param>
    /// <param name="C">channels</param>
    /// <remarks>
    /// 1.	Method Signature: The method EncoderForward is defined with parameters similar to the C function.
    /// 2.	Index Calculation: Uses index calculations to access the correct positions in the 1D arrays.
    /// 3.	Loop Structure: Nested loops iterate over batches(B), tokens(T), and the embedding dimension(C).
    /// 4.	Vector Addition: Adds the token and position embeddings and stores the result in the output array.
    /// </remarks>
    public static void EncoderForward(float[] output, int[] input, float[] wte, float[] wpe, int B, int T, int C)
    {
        // output is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
        // input is (B,T) of integers, holding the token ids at each (b,t) position
        // wte is (V,C) of token embeddings, short for "weight token embeddings"
        // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                // seek to the output position in output[b,t,:]
                int outIndex = b * T * C + t * C;
                // get the index of the token at input[b, t]
                int ix = input[b * T + t];
                // seek to the position in wte corresponding to the token
                int wteIndex = ix * C;
                // seek to the position in wpe corresponding to the position
                int wpeIndex = t * C;
                // add the two vectors and store the result in output[b,t,:]
                for (int i = 0; i < C; i++)
                {
                    output[outIndex + i] = wte[wteIndex + i] + wpe[wpeIndex + i];
                }
            }
        }
    }

    // The same as above, but with int[B,T] input, float[B,T,C] output, float[V,C] wte, float[maxT,C] wpe
    public static void EncoderForward(float[,,] output, int[,] input, float[,] wte, float[,] wpe)
    {
        Debug.Assert(input.GetLength(0) == output.GetLength(0));
        Debug.Assert(input.GetLength(1) == output.GetLength(1));
        Debug.Assert(wte.GetLength(1) == wpe.GetLength(1));
        Debug.Assert(wte.GetLength(1) == output.GetLength(2));

        int batch_size = input.GetLength(0);
        int sequence_length = input.GetLength(1);
        int channels = wte.GetLength(1);
        for (int batch_index = 0; batch_index < batch_size; batch_index++)
        {
            for (int token_index = 0; token_index < sequence_length; token_index++)
            {
                for (int channel = 0; channel < channels; channel++)
                {
                    int token_id = input[batch_index, token_index];

                    // add the two vectors and store the result in output[batch_index, token_index, channel]
                    output[batch_index, token_index, channel] = wte[token_id, channel] + wpe[token_index, channel];
                }
            }
        }
    }

    /// <param name="dwte">delta weight token embeddings</param>
    /// <param name="dwpe">delta weight positional embedding</param>
    /// <param name="doutput"></param>
    /// <param name="input">input is (B,T) of integers, holding the token ids at each (b,t) position</param>
    /// <param name="B">batch_size</param>
    /// <param name="T">sequence_length</param>
    /// <param name="C">channels</param>
    /// <remarks>
    /// 1.	Method Signature: The method EncoderBackward is defined with parameters similar to the C function.
    /// 2.	Index Calculation: Uses index calculations to access the correct positions in the 1D arrays.
    /// 3.	Loop Structure: Nested loops iterate over batches(B), tokens(T), and the embedding dimension(C).
    /// 4.	Gradient Accumulation: Adds the gradients from doutput to the corresponding positions in dwte and dwpe.
    /// </remarks>
    public static void EncoderBackward(float[] dwte, float[] dwpe, float[] doutput, int[] input, int B, int T, int C)
    {
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int doutIndex = b * T * C + t * C;
                int ix = input[b * T + t];
                int dwteIndex = ix * C;
                int dwpeIndex = t * C;
                for (int i = 0; i < C; i++)
                {
                    float d = doutput[doutIndex + i];
                    dwte[dwteIndex + i] += d;
                    dwpe[dwpeIndex + i] += d;
                }
            }
        }
    }

    // The same as above but with int[B,T] input, float[B,T,C] doutput, float[V,C] dwte, float[maxT,C] dwpe

    /// <param name="dwte">delta weight token embeddings</param>
    /// <param name="dwpe">delta weight positional embedding</param>
    public static void EncoderBackward(float[,] dwte, float[,] dwpe, float[,,] doutput, int[,] input)
    {
        Debug.Assert(input.GetLength(0) == doutput.GetLength(0));
        Debug.Assert(input.GetLength(1) == doutput.GetLength(1));
        Debug.Assert(dwte.GetLength(1) == dwpe.GetLength(1));
        Debug.Assert(dwte.GetLength(1) == doutput.GetLength(2));

        int batch_size = input.GetLength(0);
        int sequence_length = input.GetLength(1);
        int channels = dwte.GetLength(1);
        for (int batch_index = 0; batch_index < batch_size; batch_index++)
        {
            for (int token_index = 0; token_index < sequence_length; token_index++)
            {
                for (int channel = 0; channel < channels; channel++)
                {
                    int token_id = input[batch_index, token_index];
                    float d = doutput[batch_index, token_index, channel];
                    dwte[token_id, channel] += d;
                    dwpe[token_index, channel] += d;
                }
            }
        }
    }

    /// <param name="output"></param>
    /// <param name="mean"></param>
    /// <param name="rstd"></param>
    /// <param name="input"></param>
    /// <param name="weight"></param>
    /// <param name="bias"></param>
    /// <param name="B">batch_size</param>
    /// <param name="T">sequence_length</param>
    /// <param name="C">channels</param>
    /// <remarks>
    /// 1.	Method Signature: The method LayerNormForward is defined with parameters similar to the C function.
    /// 2.	Index Calculation: Uses index calculations to access the correct positions in the 1D arrays.
    /// 3.	Loop Structure: Nested loops iterate over batches(B), tokens(T), and the embedding dimension(C).
    /// 4.	Mean and Variance Calculation: Computes the mean and variance for normalization.
    /// 5.	Normalization and Scaling: Normalizes the input, scales it with weight, shifts it with bias, and writes the result to the output array.
    /// 6.	Caching: Stores the mean and reciprocal standard deviation (rstd) for use in the backward pass.
    /// </remarks>
    public static void LayerNormForward(float[] output, float[] mean, float[] rstd, float[] input, float[] weight, float[] bias, int B, int T, int C)
    {
        // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        // both input and output are (B,T,C) of the activations
        // mean and rstd (reciprocal standard deviation) are (B,T) buffers, to be used later in backward pass
        // at each position (b,t) of the input, the C-dimensional vector
        // of activations gets normalized, then scaled and shifted
        float eps = 1e-5f;
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                // seek to the input position input[b,t,:]
                int inputIndex = b * T * C + t * C;
                // calculate the mean
                float m = 0.0f;
                for (int i = 0; i < C; i++)
                {
                    m += input[inputIndex + i];
                }
                m /= C;
                // calculate the variance (without any bias correction)
                float v = 0.0f;
                for (int i = 0; i < C; i++)
                {
                    float xshift = input[inputIndex + i] - m;
                    v += xshift * xshift;
                }
                v /= C;
                // calculate the rstd (reciprocal standard deviation)
                float s = 1.0f / MathF.Sqrt(v + eps);
                // seek to the output position in output[b,t,:]
                int outputIndex = b * T * C + t * C;
                for (int i = 0; i < C; i++)
                {
                    float n = s * (input[inputIndex + i] - m); // normalize
                    float o = n * weight[i] + bias[i]; // scale and shift
                    output[outputIndex + i] = o; // write
                }
                // cache the mean and rstd for the backward pass later
                mean[b * T + t] = m;
                rstd[b * T + t] = s;
            }
        }
    }

    /// <param name="dinput"></param>
    /// <param name="dweight"></param>
    /// <param name="dbias"></param>
    /// <param name="doutput"></param>
    /// <param name="input"></param>
    /// <param name="weight"></param>
    /// <param name="mean"></param>
    /// <param name="rstd"></param>
    /// <param name="B">batch_size</param>
    /// <param name="T">sequence_length</param>
    /// <param name="C">channels</param>
    /// <remarks>
    /// 1.	Method Signature: The method LayerNormBackward is defined with parameters similar to the C function.
    /// 2.	Index Calculation: Uses index calculations to access the correct positions in the 1D arrays.
    /// 3.	Loop Structure: Nested loops iterate over batches(B), tokens(T), and the embedding dimension(C).
    /// 4.	Gradient Calculation: Computes the gradients for dinput, dweight, and dbias using the provided formulas.
    /// 5.	Accumulation: Accumulates the gradients in the respective arrays.
    /// </remarks>
    public static void LayerNormBackward(float[] dinput, float[] dweight, float[] dbias, float[] doutput, float[] input, float[] weight, float[] mean, float[] rstd, int B, int T, int C)
    {
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int indexBase = b * T * C + t * C;
                float mean_bt = mean[b * T + t];
                float rstd_bt = rstd[b * T + t];

                // first: two reduce operations
                float dnorm_mean = 0.0f;
                float dnorm_norm_mean = 0.0f;
                for (int i = 0; i < C; i++)
                {
                    float norm_bti = (input[indexBase + i] - mean_bt) * rstd_bt;
                    float dnorm_i = weight[i] * doutput[indexBase + i];
                    dnorm_mean += dnorm_i;
                    dnorm_norm_mean += dnorm_i * norm_bti;
                }
                dnorm_mean /= C;
                dnorm_norm_mean /= C;

                // now iterate again and accumulate all the gradients
                for (int i = 0; i < C; i++)
                {
                    float norm_bti = (input[indexBase + i] - mean_bt) * rstd_bt;
                    float dnorm_i = weight[i] * doutput[indexBase + i];
                    // gradient contribution to bias
                    dbias[i] += doutput[indexBase + i];
                    // gradient contribution to weight
                    dweight[i] += norm_bti * doutput[indexBase + i];
                    // gradient contribution to input
                    float dval = 0.0f;
                    dval += dnorm_i; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    dinput[indexBase + i] += dval;
                }
            }
        }
    }

    /// <param name="B">batch_size</param>
    /// <param name="T">sequence_length</param>
    /// <param name="C">channels</param>
    /// <param name="OC">OC is short for "output channels"</param>
    /// <remarks>
    /// 1.	Method Signature: The method MatmulForwardNaive is defined with parameters similar to the C function.
    /// 2.	Parallelization: Uses Parallel.For to parallelize the outer loop over batches (B).
    /// 3.	Index Calculation: Uses index calculations to access the correct positions in the 1D arrays.
    /// 4.	Matrix Multiplication: Performs the matrix multiplication and adds the bias if it is not null.
    /// 5.	Output Assignment: Stores the result in the output array.
    /// </remarks>
    public static void MatmulForwardNaive(float[] output, float[] input, float[] weight, float[] bias, int B, int T, int C, int OC)
    {
        // the most naive implementation of matrix multiplication
        // this serves as an algorithmic reference, and as a fallback for
        // unfriendly input shapes inside matmul_forward(), below.
        Parallel.For(0, B, b =>
        {
            for (int t = 0; t < T; t++)
            {
                int bt = b * T + t;
                for (int o = 0; o < OC; o++)
                {
                    float val = (bias != null) ? bias[o] : 0.0f;
                    for (int i = 0; i < C; i++)
                    {
                        val += input[bt * C + i] * weight[o * C + i];
                    }
                    output[bt * OC + o] = val;
                }
            }
        });
    }

    /// <param name="B">batch_size</param>
    /// <param name="T">sequence_length</param>
    /// <param name="C">channels</param>
    /// <param name="OC">OC is short for "output channels"</param>
    /// <remarks>
    /// 1.	Method Signature: The method MatmulForward is defined with parameters similar to the C function.
    /// 2.	Fallback to Naive Version: Checks if the loop unrolling condition is met; if not, it calls the naive version.
    /// 3.	Parallelization: Uses Parallel.For to parallelize the outer loop over batches (B) and tokens (T).
    /// 4.	Loop Unrolling: Unrolls the loop to improve performance by reusing loaded weights.
    /// 5.	Index Calculation: Uses index calculations to access the correct positions in the 1D arrays.
    /// 6.	Matrix Multiplication: Performs the matrix multiplication and adds the bias if it is not null.
    /// 7.	Output Assignment: Stores the result in the output array.
    /// </remarks>
    public static void MatmulForward(float[] output, float[] input, float[] weight, float[] bias, int B, int T, int C, int OC)
    {
        // most of the running time is spent here and in matmul_backward
        // therefore, the implementation below is very mildly optimized
        // this function is otherwise identical to that of matmul_forward_naive()
        // OC is short for "output channels"
        // input is (B,T,C), weight is (OC, C), bias is (OC)
        // output will be (B,T,OC)

        // make sure the tiled loop will be correct or fallback to naive version
        const int LOOP_UNROLL = 8;
        if (B * T % LOOP_UNROLL != 0)
        {
            MatmulForwardNaive(output, input, weight, bias, B, T, C, OC);
            return;
        }

        // collapse the B and T loops into one and turn it into a strided loop.
        // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
        Parallel.For(0, B * T / LOOP_UNROLL, obt =>
        {
            for (int o = 0; o < OC; o++)
            {
                // we'll keep LOOP_UNROLL many results in registers
                float[] result = new float[LOOP_UNROLL];
                // initialize the bias, if it exists
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++)
                {
                    result[ibt] = (bias != null) ? bias[o] : 0.0f;
                }
                // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
                // the value of weight[i + o * C] and reuse it.
                for (int i = 0; i < C; i++)
                {
                    float w = weight[i + o * C];
                    for (int ibt = 0; ibt < LOOP_UNROLL; ibt++)
                    {
                        int bt = obt * LOOP_UNROLL + ibt;
                        result[ibt] += input[bt * C + i] * w;
                    }
                }
                // write back results to main memory
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++)
                {
                    int bt = obt * LOOP_UNROLL + ibt;
                    output[bt * OC + o] = result[ibt];
                }
            }
        });
    }

    /// <param name="B">batch_size</param>
    /// <param name="T">sequence_length</param>
    /// <param name="C">channels</param>
    /// <param name="OC">OC is short for "output channels"</param>
    /// <remarks>
    /// 1.	Method Signature: The method MatmulBackward is defined with parameters similar to the C function.
    /// 2.	Parallelization: Uses Parallel.For to parallelize the outer loops over batches (B) and output channels (OC).
    /// 3.	Index Calculation: Uses index calculations to access the correct positions in the 1D arrays.
    /// 4.	Backward Pass into Input: Computes the gradient with respect to the input (dinput).
    /// 5.	Backward Pass into Weight/Bias: Computes the gradient with respect to the weight (dweight) and bias (dbias).
    /// 6.	Conditional Bias Update: Updates the bias only if it is not null.
    /// </remarks>
    public static void MatmulBackward(float[] dinput, float[] dweight, float[] dbias, float[] doutput, float[] input, float[] weight, int B, int T, int C, int OC)
    {
        // most of the running time is spent here and in matmul_forward
        // this backward could be done in a single "round" of loops
        // but that doesn't afford an efficient parallelization strategy

        // backward into input first, parallelize over B,T
        Parallel.For(0, B, b =>
        {
            for (int t = 0; t < T; t++)
            {
                int doutIndex = b * T * OC + t * OC;
                int dinpIndex = b * T * C + t * C;
                for (int o = 0; o < OC; o++)
                {
                    int wrowIndex = o * C;
                    float d = doutput[doutIndex + o];
                    for (int i = 0; i < C; i++)
                    {
                        dinput[dinpIndex + i] += weight[wrowIndex + i] * d;
                    }
                }
            }
        });

        // backward into weight/bias, parallelize over output channels OC
        Parallel.For(0, OC, o =>
        {
            for (int b = 0; b < B; b++)
            {
                for (int t = 0; t < T; t++)
                {
                    int doutIndex = b * T * OC + t * OC;
                    int inpIndex = b * T * C + t * C;
                    int dwrowIndex = o * C;
                    float d = doutput[doutIndex + o];
                    if (dbias != null)
                    {
                        dbias[o] += d;
                    }
                    for (int i = 0; i < C; i++)
                    {
                        dweight[dwrowIndex + i] += input[inpIndex + i] * d;
                    }
                }
            }
        });
    }

    /// <param name="B">batch_size</param>
    /// <param name="T">sequence_length</param>
    /// <param name="C">channels</param>
    /// <param name="NH">number of heads</param>
    public static void AttentionForward(float[] output, float[] preatt, float[] att, float[] input, int B, int T, int C, int NH)
    {
        // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (B, T, C)
        // attention is the only layer that mixes information across time
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / MathF.Sqrt(hs);

        Parallel.For(0, B, b =>
        {
            for (int t = 0; t < T; t++)
            {
                for (int h = 0; h < NH; h++)
                {
                    int queryIndex = b * T * C3 + t * C3 + h * hs;
                    int preattIndex = b * NH * T * T + h * T * T + t * T;
                    int attIndex = b * NH * T * T + h * T * T + t * T;

                    // pass 1: calculate query dot key and maxval
                    float maxval = -10000.0f; // TODO something better
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        int keyIndex = b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                        // (query_t) dot (key_t2)
                        float val = 0.0f;
                        for (int i = 0; i < hs; i++)
                        {
                            val += input[queryIndex + i] * input[keyIndex + i];
                        }
                        val *= scale;
                        if (val > maxval)
                        {
                            maxval = val;
                        }

                        preatt[preattIndex + t2] = val;
                    }

                    // pass 2: calculate the exp and keep track of sum
                    // maxval is being calculated and subtracted only for numerical stability
                    float expsum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        float expv = MathF.Exp(preatt[preattIndex + t2] - maxval);
                        expsum += expv;
                        att[attIndex + t2] = expv;
                    }
                    float expsumInv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                    // pass 3: normalize to get the softmax
                    for (int t2 = 0; t2 < T; t2++)
                    {
                        if (t2 <= t)
                        {
                            att[attIndex + t2] *= expsumInv;
                        }
                        else
                        {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            att[attIndex + t2] = 0.0f;
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    int outIndex = b * T * C + t * C + h * hs;
                    for (int i = 0; i < hs; i++) { output[outIndex + i] = 0.0f; }
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        int valueIndex = b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                        float attValue = att[attIndex + t2];
                        for (int i = 0; i < hs; i++)
                        {
                            output[outIndex + i] += attValue * input[valueIndex + i];
                        }
                    }
                }
            }
        });
    }

}
