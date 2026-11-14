#define CHUNKS ((HEAD_DIM + 31) / 32)

extern "C" __global__
void attention_delta(const float* O, const float* G, float* Delta, int rows, int d)
{
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;

    long long base = (long long) row * d;
    float value = 0.0f;
    for (int r = 0; r < (d +31) / 32; ++r) 
    {
        int feature = lane + 32*r;
        if (feature < d) value += G[base + feature] * O[base + feature];
    }

    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_xor_sync(0xffffffff, value, offset);

    if (lane == 0) Delta[row] = value;
}

extern "C" __global__
void attention_dq(
    const float* Q, const float* K, const float* V, const float* G, 
    const float *LSE, const float* Delta, float* DQ,
    int H, int Tq, int Tk, int past_len,
    long long qb, long long qh, long long qt, long long qd,
    long long kb, long long kh, long long kt, long long kd,
    long long vb, long long vh, long long vt, long long vd)
{
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    int b = blockIdx.y / H;
    int h = blockIdx.y % H;

    int query_start = blockIdx.x * 8;
    int query = query_start + warp;

    int limit = min(Tk, past_len + min(query_start + 7, Tq - 1) + 1);

    long long row = ((long long) b * H + h) * Tq + query;
    float scale = rsqrtf((float)HEAD_DIM);

    __shared__ float keys[KEY_TILE][HEAD_DIM];
    __shared__ float values[KEY_TILE][HEAD_DIM];

    float q[CHUNKS], g[CHUNKS], acc[CHUNKS];

    #pragma unroll
    for (int r = 0; r < CHUNKS; ++r) {
        int feature = lane + 32*r;
        q[r] = query < Tq && feature < HEAD_DIM
            ? Q[b*qb + h*qh + query*qt + feature*qd] : 0.0f;
        g[r] = query < Tq && feature < HEAD_DIM
            ? G[row*HEAD_DIM + feature] : 0.0f;
        acc[r] = 0.0f;
    }

    float lse = query < Tq ? LSE[row] : 0.0f;
    float delta = query < Tq ? Delta[row] : 0.0f;

    for (int start = 0; start < limit; start += KEY_TILE)
    {
        for (int p = threadIdx.x; p < KEY_TILE*HEAD_DIM; p += blockDim.x) 
        {
            int token = p / HEAD_DIM;
            int feature = p % HEAD_DIM;
            int key = start + token;
            keys[token][feature] = key < limit ? K[b*kb + h*kh + key*kt + feature*kd] : 0.0f;
            values[token][feature] = key < limit ? V[b*vb + h*vh + key*vt + feature*vd] : 0.0f;
        }
        __syncthreads();

        if (query < Tq) 
        {
            for (int token = 0; token < KEY_TILE && start + token < Tk && start + token <= past_len + query; ++token) 
            {
                float score = 0.0f;
                float dp = 0.0f;

                #pragma unroll
                for (int r = 0; r < CHUNKS; ++r) 
                {
                    int feature = lane + 32*r;
                    if (feature < HEAD_DIM)
                    {
                        score += q[r] * keys[token][feature];
                        dp += g[r] * values[token][feature];
                    }
                }
                for (int offset = 16; offset > 0; offset /= 2)
                {
                    score += __shfl_xor_sync(0xffffffff, score, offset);
                    dp += __shfl_xor_sync(0xffffffff, dp, offset);
                }
                score *= scale;

                float weight = expf(score - lse);
                float ds = weight * (dp - delta);

                #pragma unroll
                for (int r = 0; r < CHUNKS; ++r) 
                {
                    int feature = lane + 32*r;
                    if (feature < HEAD_DIM)
                        acc[r] += ds * keys[token][feature] * scale;
                }
            }
        }
        __syncthreads();
    }
    if (query < Tq) 
    {
        #pragma unroll
        for (int r = 0; r < CHUNKS; ++r) 
        {
            int feature = lane + 32*r;
            if (feature < HEAD_DIM)
                DQ[row*HEAD_DIM + feature] = acc[r];
        }
    }
}

extern "C" __global__
void attention_dkdv(
    const float* Q, const float* K, const float* V,
    const float* G, const float* LSE, const float* Delta, float* DK, float* DV,
    int H, int Tq, int Tk, int past_len,
    long long qb, long long qh, long long qt, long long qd,
    long long kb, long long kh, long long kt, long long kd,
    long long vb, long long vh, long long vt, long long vd)
{
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    int b = blockIdx.y / H;
    int h = blockIdx.y % H;

    int key_start = blockIdx.x * 8;
    int key = key_start + warp;
    int first_query = max(0, key_start - past_len);

    long long row = ((long long)b * H + h) * Tk + key;
    long long query_base = ((long long)b * H + h) * Tq;
    float scale = rsqrtf((float)HEAD_DIM);

    __shared__ float queries[KEY_TILE][HEAD_DIM];
    __shared__ float grads[KEY_TILE][HEAD_DIM];

    float k[CHUNKS], v[CHUNKS], dk[CHUNKS], dv[CHUNKS];

    #pragma unroll
    for (int r = 0; r < CHUNKS; ++r) 
    {
        int feature = lane + 32*r;
        k[r] = key < Tk && feature < HEAD_DIM ? K[b*kb + h*kh + key*kt + feature*kd] : 0.0f;
        v[r] = key < Tk && feature < HEAD_DIM ? V[b*vb + h*vh + key*vt + feature*vd] : 0.0f;
        dk[r] = 0.0f;
        dv[r] = 0.0f;
    }

    for (int start = first_query; start < Tq; start += KEY_TILE) 
    {
        for (int p = threadIdx.x; p < KEY_TILE*HEAD_DIM; p += blockDim.x) 
        {
            int token = p / HEAD_DIM;
            int feature = p % HEAD_DIM;
            int query = start + token;
            queries[token][feature] = query < Tq ? Q[b*qb + h*qh + query*qt + feature*qd] : 0.0f;
            grads[token][feature] = query < Tq ? G[(query_base + query)*HEAD_DIM + feature] : 0.0f;
        }
        __syncthreads();

        if (key < Tk) 
        {
            for (int token = 0; token < KEY_TILE && start + token < Tq; ++token) 
            {
                int query = start + token;
                if (key > past_len + query) continue;
                float score = 0.0f;
                float dp = 0.0f;
                #pragma unroll
                for (int r = 0; r < CHUNKS; ++r) 
                {
                    int feature = lane + 32*r;
                    if (feature < HEAD_DIM) 
                    {
                        score += queries[token][feature] * k[r];
                        dp += grads[token][feature] * v[r];
                    }
                }
                for (int offset = 16; offset > 0; offset /= 2)
                {
                    score += __shfl_xor_sync(0xffffffff, score, offset);
                    dp += __shfl_xor_sync(0xffffffff, dp, offset);
                }
                score *= scale;

                float probability = expf(score - LSE[query_base + query]);
                float ds = probability * (dp - Delta[query_base + query]);

                #pragma unroll
                for (int r = 0; r < CHUNKS; ++r) 
                {
                    int feature = lane + 32*r;
                    if (feature < HEAD_DIM) 
                    {
                        dk[r] += ds * queries[token][feature] * scale;
                        dv[r] += probability * grads[token][feature];
                    }
                }
            }
        }
        __syncthreads();
    }
    if (key < Tk) 
    {
        #pragma unroll
        for (int r = 0; r < CHUNKS; ++r) 
        {
            int feature = lane + 32*r;
            if (feature < HEAD_DIM) {
                DK[row*HEAD_DIM + feature] = dk[r];
                DV[row*HEAD_DIM + feature] = dv[r];
            }
        }
    }
}
