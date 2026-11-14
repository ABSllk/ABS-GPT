#define CHUNKS ((HEAD_DIM + 31) / 32)

extern "C" __global__
void attention_forward(
    const float* Q, const float* K, const float* V, float* O, float* LSE,
    int H, int Tq, int Tk, int past_len,
    long long qb, long long qh, long long qt, long long qd,
    long long kb, long long kh, long long kt, long long kd,
    long long vb, long long vh, long long vt, long long vd)
{
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int b = blockIdx.y / H;
    const int h = blockIdx.y % H;
    const int query_start = blockIdx.x * 8;
    const int query = query_start + warp;
    const int limit = min(Tk, past_len + min(query_start + 7, Tq - 1) + 1);

    __shared__ float keys[KEY_TILE][HEAD_DIM];
    __shared__ float values[KEY_TILE][HEAD_DIM];
    float q[CHUNKS];
    float acc[CHUNKS];
    #pragma unroll
    for (int r = 0; r < CHUNKS; ++r) 
    {
        int feature = lane + 32*r;
        q[r] = query < Tq && feature < HEAD_DIM
            ? Q[b*qb + h*qh + query*qt + feature*qd] : 0.0f;
        acc[r] = 0.0f;
    }
    float m = -1.0f / 0.0f;
    float ell = 0.0f;

    for (int start = 0; start < limit; start += KEY_TILE) 
    {
        // All threads cooperate to load this K/V tile.
        for (int p = threadIdx.x; p < KEY_TILE*HEAD_DIM; p += blockDim.x) 
        {
            int token = p / HEAD_DIM;
            int feature = p % HEAD_DIM;
            int key = start + token;
            keys[token][feature] = key < limit ? K[b*kb + h*kh + key*kt + feature*kd] : 0.0f;
            values[token][feature] = key < limit ? V[b*vb + h*vh + key*vt + feature*vd] : 0.0f;
        }
        __syncthreads();

        // The branch is uniform within each warp. Tail warps still reach barriers.
        if (query < Tq) 
        {
            for (int token = 0; token < KEY_TILE && start + token < Tk && start + token <= past_len + query; ++token) 
            {
                float score = 0.0f;
                #pragma unroll
                for (int r = 0; r < CHUNKS; ++r) 
                {
                    int feature = lane + 32*r;
                    if (feature < HEAD_DIM)
                        score += q[r] * keys[token][feature];
                }
                for (int offset = 16; offset > 0; offset /= 2)
                    score += __shfl_xor_sync(0xffffffff, score, offset);
                score *= rsqrtf((float)HEAD_DIM);

                float next_m = fmaxf(m, score);
                float alpha = expf(m - next_m);
                float weight = expf(score - next_m);
                ell = alpha * ell + weight;
                #pragma unroll
                for (int r = 0; r < CHUNKS; ++r) 
                {
                    int feature = lane + 32*r;
                    if (feature < HEAD_DIM)
                        acc[r] = alpha * acc[r] + weight * values[token][feature];
                }
                m = next_m;
            }
        }
        __syncthreads();  // Readers finish before the next tile overwrites shared memory.
    }

    if (query < Tq) 
    {
        long long row = ((long long)b * H + h) * Tq + query;
        #pragma unroll
        for (int r = 0; r < CHUNKS; ++r) 
        {
            int feature = lane + 32*r;
            if (feature < HEAD_DIM)
                O[row*HEAD_DIM + feature] = acc[r] / ell;
        }
        if (lane == 0)
            LSE[row] = m + logf(ell);
    }
}
