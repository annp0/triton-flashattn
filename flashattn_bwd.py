import torch
import triton
import triton.language as tl

configs = [
    triton.Config({'BLOCK_MAJOR': 64, 'BLOCK_MINOR': 16}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_MAJOR': 64, 'BLOCK_MINOR': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_MAJOR': 128, 'BLOCK_MINOR': 128}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_MAJOR': 128, 'BLOCK_MINOR': 32}, num_stages=2, num_warps=8),
    triton.Config({'BLOCK_MAJOR': 128, 'BLOCK_MINOR': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_MAJOR': 256, 'BLOCK_MINOR': 64}, num_stages=3, num_warps=8),
]


@triton.autotune(configs=[
    triton.Config({'BLOCK_MAJOR': 32}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_MAJOR': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_MAJOR': 128}, num_stages=3, num_warps=8),
], key=['N', 'HEAD_DIM'])
@triton.jit
def _attn_bwd_prep(
    O, DO,
    Delta,
    Z, H, N,
    HEAD_DIM: tl.constexpr,
    BLOCK_MAJOR: tl.constexpr
):
    BLOCK_M: tl.constexpr = BLOCK_MAJOR
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, HEAD_DIM)
    o = tl.load(O + (off_hz * N + offs_m[:, None]) * HEAD_DIM + offs_n[None, :])
    do = tl.load(DO + (off_hz * N + offs_m[:, None]) * HEAD_DIM + offs_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N + offs_m, delta)

@triton.jit
def _attn_bwd_dkdv(
    on_band: tl.constexpr,
    dk, dv,
    Q, k, v, scale,
    DO, B, D,
    start_n, start_m, num_steps,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_col = tl.arange(0, HEAD_DIM)

    # load in qT at row-tile m / HEAD_DIM
    ptr_qT = Q + offs_m[None, :] * HEAD_DIM + offs_col[:, None]
    ptr_do = DO + offs_m[:, None] * HEAD_DIM + offs_col[None, :]
    curr_m = start_m
    # for dV, we need corresponding DO and DT
    # hold n on dK, dV, slide m on Q, DO
    for idx in range(num_steps):
        qT = tl.load(ptr_qT) 
        # slide down offs_m
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        b = tl.load(B + offs_m)
        # first, calculate DT. use the exp2 trick again
        kTq = tl.dot(k, qT) * scale * 1.4426950408889634
        dT = tl.math.exp2(kTq - b[None, :])
        if on_band:
            # need m >= n (just recalculate DT)
            # we can mask on dT directly since we know the sum already
            mask = (offs_m[None, :] >= offs_n[:, None])
            dT = tl.where(mask, dT, 0.0)
        do = tl.load(ptr_do)
        dT_f16 = dT.to(tl.float16)
        dv += tl.dot(dT_f16, do)
        delta = tl.load(D + offs_m)
        # dD, no accumulation needed since we have full head_dim
        ddT = tl.dot(v, tl.trans(do)).to(tl.float32)
        daT = dT * (ddT - delta[None, :])
        daT = daT.to(tl.float16)
        dk += tl.dot(daT, tl.trans(qT))

        curr_m += BLOCK_M1
        ptr_qT += BLOCK_M1 * HEAD_DIM
        ptr_do += BLOCK_M1 * HEAD_DIM
    
    # THIS IS WRONG BECAUSE THE FUNCTION IS CALLED TWICE!!!
    # dk = dk * scale

    return dk, dv 
        
@triton.jit
def _attn_bwd_dq(
    on_band: tl.constexpr,
    dq, q, K, V, scale,
    do, b, D,
    start_m, start_n, num_steps,
    BLOCK_M2: tl.constexpr, 
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr
): 
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_col = tl.arange(0, HEAD_DIM)
    ptr_vT = V + offs_n[None, :] * HEAD_DIM + offs_col[:, None]
    ptr_kT = K + offs_n[None, :] * HEAD_DIM + offs_col[:, None]
    delta = tl.load(D + offs_m)
    curr_n = start_n
    for idx in range(num_steps):
        kT = tl.load(ptr_kT)
        vT = tl.load(ptr_vT)
        qk = tl.dot(q, kT)
        qk = qk * scale * 1.4426950408889634
        d = tl.math.exp2(qk - b)
        if on_band:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None]) >= (offs_n[None, :])
            d = tl.where(mask, d, 0.0)
        dd = tl.dot(do, vT).to(tl.float32)
        da = d * (dd - delta[:, None])
        da = da.to(tl.float16)
        dq += tl.dot(da, tl.trans(kT)) * scale
        curr_n += BLOCK_N2
        ptr_kT += BLOCK_N2 * HEAD_DIM
        ptr_vT += BLOCK_N2 * HEAD_DIM
    
    return dq

@triton.autotune(configs=configs, key=['N', 'HEAD_DIM'])
@triton.jit
def _attn_bwd(
    Q, K, V, scale,
    DO, DQ, DK, DV,
    B, D,
    Z, H, N,
    HEAD_DIM: tl.constexpr,
    BLOCK_MAJOR: tl.constexpr,
    BLOCK_MINOR: tl.constexpr
):
    BLOCK_M1: tl.constexpr = BLOCK_MINOR
    BLOCK_N1: tl.constexpr = BLOCK_MAJOR
    BLOCK_M2: tl.constexpr = BLOCK_MAJOR
    BLOCK_N2: tl.constexpr = BLOCK_MINOR
    # assumptions, so that I don't need to think about masks
    # for dK, dV, we fix n and scan m
    # for dQ, we fix m and scan n 
    tl.static_assert(BLOCK_MAJOR % BLOCK_MINOR == 0)
    
    pid = tl.program_id(0)
    pid_hz = tl.program_id(1)
    head_start = pid_hz * N
    off_head_start_off = head_start * HEAD_DIM

    # move pointers to head
    Q += off_head_start_off
    K += off_head_start_off
    V += off_head_start_off
    DO += off_head_start_off
    DQ += off_head_start_off
    DK += off_head_start_off
    DV += off_head_start_off
    B += head_start
    D += head_start

    offs_col = tl.arange(0, HEAD_DIM)

    # for dk, dv, move to the corresponding row_tile
    start_n = pid * BLOCK_N1
    # for O, moving to corresponding starting position
    # causal masking: on-band: start_n -> + BLOCK_N1
    #                off-band: + BLOCK_N1 -> N
    start_m = start_n

    # offsets for loading n
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    # output tiles
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load corresponding tiles for K and V
    # they stay constant
    k = tl.load(K + offs_n[:, None] * HEAD_DIM + offs_col[None, :])
    v = tl.load(V + offs_n[:, None] * HEAD_DIM + offs_col[None, :])

    num_steps = BLOCK_N1 // BLOCK_M1

    dk, dv = _attn_bwd_dkdv(1, dk, dv,
                            Q, k, v, scale,
                            DO, B, D,
                            start_n, start_m, num_steps,
                            BLOCK_M1, BLOCK_N1,
                            HEAD_DIM)

    start_m += BLOCK_N1
    num_steps = (N - start_m) // BLOCK_M1

    dk, dv = _attn_bwd_dkdv(0, dk, dv,
                            Q, k, v, scale,
                            DO, B, D,
                            start_n, start_m, num_steps,
                            BLOCK_M1, BLOCK_N1,
                            HEAD_DIM)

    tl.store(DV + offs_n[:, None] * HEAD_DIM + offs_col, dv)
    tl.store(DK + offs_n[:, None] * HEAD_DIM + offs_col, dk * scale)

    # on-band dq
    # for dq, in this row-tile. uses the same dO
    start_m = pid * BLOCK_M2
    # for K, V, we start from start_m and end at one BLOCK_M2 over: 
    end_n = start_m + BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    # prepare values for dq, those values are fixed
    q = tl.load(Q + offs_m[:, None] * HEAD_DIM + offs_col)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * HEAD_DIM + offs_col)
    b = tl.load(B + offs_m)[:, None]
    num_steps = BLOCK_M2 // BLOCK_N2

    dq = _attn_bwd_dq(1,
                      dq, q, K, V, scale,
                      do, b, D, 
                      start_m, start_n, num_steps,
                      BLOCK_M2, BLOCK_N2, HEAD_DIM)

    # for the 2nd stage, we go from end_n to 0 for K, V
    end_n -= BLOCK_M2

    num_steps = end_n // BLOCK_N2

    dq = _attn_bwd_dq(0,
                      dq, q, K, V, scale,
                      do, b, D, 
                      start_m, 0, num_steps,
                      BLOCK_M2, BLOCK_N2, HEAD_DIM)

    tl.store(DQ + offs_m[:, None] * HEAD_DIM + offs_col[None, :], dq)