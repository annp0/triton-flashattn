import torch
import triton
import triton.language as tl

# not on blackwell / hopper, using raw ptrs (not descriptors), no FP8
# my triton is old - descriptors are still experimental!

# took a lot of time to debug the fwd kernel, slept at 3AM yesterday
# hope backprop is faster

configs = [
    triton.Config({'BLOCK_MAJOR': 64, 'BLOCK_MINOR': 16}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_MAJOR': 64, 'BLOCK_MINOR': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_MAJOR': 128, 'BLOCK_MINOR': 128}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_MAJOR': 128, 'BLOCK_MINOR': 32}, num_stages=2, num_warps=8),
    triton.Config({'BLOCK_MAJOR': 128, 'BLOCK_MINOR': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_MAJOR': 256, 'BLOCK_MINOR': 64}, num_stages=3, num_warps=8),
]

@triton.jit
def _attn_fwd_kv_loop(
    # putting variables in scope...
    on_band: tl.constexpr,
    low, high, head_start, scale,
    q, ptr_k, ptr_v,
    offs_n: tl.constexpr, offs_row_q, offs_col: tl.constexpr,
    max_r, expsum_r, output,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr
):
    # for each row-tile of Q, loop through the K,V of tokens in range(low, high)
    for off_n in tl.range(low, high, BLOCK_N):
        # offsets of kv within a head
        offs_row_kv = off_n + offs_n
        k = tl.load(ptr_k + (head_start + offs_row_kv)[:, None] * HEAD_DIM + offs_col)
        v = tl.load(ptr_v + (head_start + offs_row_kv)[:, None] * HEAD_DIM + offs_col)
        k = k.T
        # dot product of Q [pid_m] x K [low - high]
        qk = tl.dot(q, k) * scale
        if on_band:
            causal_mask = (offs_row_q[:, None]) >= (offs_row_kv[None, :])
            qk = qk + tl.where(causal_mask, 0, -1.0e9)
        max_r_now = tl.maximum(max_r, tl.max(qk, 1))
        qk = qk - max_r_now[:, None]
        exp_qk = tl.math.exp2(qk)
        alpha = tl.math.exp2(max_r - max_r_now)
        # sum across columns - row-wise sum
        expsum_r_now = tl.sum(exp_qk, 1)
        
        output = output * (alpha[:, None])

        exp_qk = exp_qk.to(tl.float16)
        # for every entry, compute and add weight-avged values
        # use tensor cores: exp_qk(f16), v(f16), output(f32)
        output = tl.dot(exp_qk, v, output)

        expsum_r = expsum_r * alpha + expsum_r_now
        max_r = max_r_now
    return output, expsum_r, max_r

@triton.autotune(configs=configs, key=['N', 'HEAD_DIM'])
@triton.jit
def _attn_fwd (
    scale, # scaling factor 1/sqrt(d_k)
    ptr_b, # output tensor to store \log\sum_j\exp(A_{ij}) per row 
            # to be used in backward 
            # dimension: (Z * H * N, 1)
    Z, # batch size
    H, # number of heads
    N, # number of tokens
    ptr_q, # pointer to Q (Z * H * N, HEAD_DIM)
            # each row of Q corresponds to a query from a specific token in a specific head & batch
    ptr_k, # pointer to K
    ptr_v, # pointer to V (d_v = d_k)
    ptr_o, # pointer to O
    HEAD_DIM: tl.constexpr, # d^h_k
    BLOCK_MAJOR: tl.constexpr, # tile size in query direction
    BLOCK_MINOR: tl.constexpr, # ... in token sequence direction
):
    # to avoid thinking about masking, I assume:
    # 1. N is a multiple of BLOCK_M (checked outside, since N is not constexpr)
    # 2. BLOCK_N is a multiple of BLOCK_M
    BLOCK_M: tl.constexpr = BLOCK_MAJOR
    BLOCK_N: tl.constexpr = BLOCK_MINOR
    tl.static_assert(BLOCK_M % BLOCK_N == 0) 

    pid_m = tl.program_id(0) # row-tile block-id (which BLOCK_M of the Query for a specific batch & head)
    pid_hz = tl.program_id(1) # which batch and head we are in
    # we could use a 3D launch grid, but 2D might have slightly less overhead for triton
    # batch & head id
    pid_z = pid_hz // H
    pid_h = pid_hz % H

    # the range of the current head
    head_start = pid_z * (H * N) + pid_h * N
    # head_end = head_start + N

    # row offset: from head_start, moving down pid_m blocks
    off_m = pid_m * BLOCK_M
    offs_m = tl.arange(0, BLOCK_M)
    # offsets in q within a head
    offs_row_q = off_m + offs_m

    # initialize running statistics (sftmax) in SRAM / Registers
    max_r = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    expsum_r = tl.zeros([BLOCK_M], dtype=tl.float32)
    output = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # becasue we use powers of 2 (faster) and exp(e) = 2^(x * log_2e)
    scale = scale * 1.44269504

    # recompute offs_row adhoc might reduce register pressure
    # offs_row (head_start + off_m + offs_m)
    
    offs_col = tl.arange(0, HEAD_DIM) # shape (HEAD_DIM,)
    
    q = tl.load(ptr_q + (head_start + offs_row_q)[:, None] * HEAD_DIM + offs_col[None, :])

    offs_n = tl.arange(0, BLOCK_N)

    # when off-band...
    low, high = 0, pid_m * BLOCK_M
    output, expsum_r, max_r= _attn_fwd_kv_loop(0, low, high, 
                    head_start, scale,
                    q, ptr_k, ptr_v,
                    offs_n, offs_row_q, offs_col, 
                    max_r, expsum_r, output,
                    HEAD_DIM, BLOCK_N)

    # when on-band..
    low, high = pid_m * BLOCK_M, tl.minimum((pid_m + 1) * BLOCK_M, N)
    output, expsum_r, max_r = _attn_fwd_kv_loop(1, low, high, head_start, scale,
                    q, ptr_k, ptr_v,
                    offs_n, offs_row_q, offs_col, 
                    max_r, expsum_r, output,
                    HEAD_DIM, BLOCK_N)

    output = output / expsum_r[:, None]
    # B stores log(sum exp)
    tl.store(ptr_b + (head_start + offs_row_q), max_r + tl.math.log2(expsum_r)) 
    tl.store(ptr_o + (head_start + offs_row_q)[:, None] * HEAD_DIM + offs_col[None, :], output.to(tl.float16))