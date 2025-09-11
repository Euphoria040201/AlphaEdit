# alphaedit_solvers.py
from typing import Optional
import torch

def update_projector_from_P_and_K(P0: torch.Tensor, Knew: torch.Tensor, tol: float = 1e-10):
    assert P0.dim() == 2 and P0.size(0) == P0.size(1), "P0 must be square (d×d)"
    assert Knew.dim() == 2 and Knew.size(0) == P0.size(0), "Knew must be d×r"

    device, dtype = P0.device, P0.dtype
    Knew = Knew.to(device=device, dtype=dtype)

    d = P0.size(0)
    R = P0 @ Knew

    if R.numel() == 0 or R.shape[1] == 0:
        P_new  = P0
        Q_add  = torch.zeros((d, 0), device=device, dtype=dtype)
        return P_new, Q_add

    U, S, Vh = torch.linalg.svd(R, full_matrices=False)

    eps  = torch.finfo(S.dtype).eps
    base = S[0] if S.numel() > 0 else torch.tensor(0.0, device=device, dtype=dtype)
    thr  = max(R.shape) * eps * base
    r    = int((S > torch.maximum(thr, torch.tensor(tol, device=device, dtype=dtype))).sum().item())

    if r == 0:
        P_new = P0
        Q_add = torch.zeros((d, 0), device=device, dtype=dtype)
    else:
        Q_add = U[:, :r]
        P_new = P0 - Q_add @ Q_add.transpose(-2, -1)
        # 如需更稳对称性：
        # P_new = 0.5 * (P_new + P_new.transpose(-2, -1))

    return P_new, Q_add

def solve_update_lowrank_cholesky_and_update_P(
    P_i: torch.Tensor,   # d×d
    K_t: torch.Tensor,   # d×r
    resid: torch.Tensor, # d×r
    lam: float,
) -> torch.Tensor:
    # 统一到 P_i 的 device/dtype
    K_t   = K_t.to(device=P_i.device, dtype=P_i.dtype)
    resid = resid.to(device=P_i.device, dtype=P_i.dtype)

    Y   = P_i @ K_t
    S0  = K_t.T @ Y
    r   = S0.size(0)
    Ir  = torch.eye(r, device=S0.device, dtype=S0.dtype)
    eps = 1e-6 if S0.dtype == torch.float32 else 1e-12

    L = torch.linalg.cholesky(S0 + lam * Ir + eps * Ir)
    B = torch.cholesky_solve(Y.T, L)

    upd_matrix = resid @ B  # d×d

    # projector 原位更新（保持在当前 device）
    P_new, _ = update_projector_from_P_and_K(P_i, K_t)
    P_i.copy_(P_new)

    return upd_matrix


def solve_update_full_and_update_P(
    P_i: torch.Tensor,   # d×d
    K_t: torch.Tensor,   # d×r
    resid: torch.Tensor, # d×r
    lam: float,
) -> torch.Tensor:
    # 统一到 P_i 的 device/dtype
    K_t   = K_t.to(device=P_i.device, dtype=P_i.dtype)
    resid = resid.to(device=P_i.device, dtype=P_i.dtype)

    d  = K_t.shape[0]
    Id = torch.eye(d, device=P_i.device, dtype=P_i.dtype)

    A = P_i @ (K_t @ K_t.T) + lam * Id
    B = P_i @ K_t @ resid.T

    upd_matrix = torch.linalg.solve(A, B)

    P_new, _ = update_projector_from_P_and_K(P_i, K_t)
    P_i.copy_(P_new)

    return upd_matrix

# ---- 按论文公式（无 K_t；使用 K_p 与 K_1）----
def alphaedit_formula_update_timed(
    P_i: torch.Tensor,                 # d×d
    resid: torch.Tensor,               # d×r1（当前编辑对应的 R）
    K_p: Optional[torch.Tensor] = None,# d×r_p（之前 1..t-1 次的键）
    K_1: Optional[torch.Tensor] = None,# d×r1（当前编辑的键）
    sync_cuda: bool = True,
    print_timing: bool = True,
) -> torch.Tensor:
    """
    Δ = R K_1^T P ( K_p K_p^T P + K_1 K_1^T P + I )^{-1}
    用 torch.linalg.solve 避免显式求逆：
      设 C = R K_1^T P，M = K_p K_p^T P + K_1 K_1^T P + I
      解 (M^T) X = C^T，返回 Δ = X^T
    不修改 P、不做近似。
    """
    from time import perf_counter
    with torch.no_grad():
        device, dtype = P_i.device, P_i.dtype
        resid = resid.to(device=device, dtype=dtype)
        if K_p is not None: K_p = K_p.to(device=device, dtype=dtype)
        if K_1 is not None: K_1 = K_1.to(device=device, dtype=dtype)

        d = P_i.shape[0]
        I = torch.eye(d, device=device, dtype=dtype)

        # 构造 M = I + K_p K_p^T P + K_1 K_1^T P
        M = I
        if (K_p is not None) and (K_p.numel() > 0):
            M = M + K_p @ (K_p.transpose(-2, -1) @ P_i)
        if (K_1 is not None) and (K_1.numel() > 0):
            M = M + K_1 @ (K_1.transpose(-2, -1) @ P_i)

        # 构造 C = R K_1^T P；若当前无键，Δ=0
        if (K_1 is None) or (K_1.numel() == 0):
            return torch.zeros((d, d), device=device, dtype=dtype)

        C = resid @ (K_1.transpose(-2, -1) @ P_i)   # d×d

        # solve： (M^T) X = C^T  ->  Δ = X^T
        if sync_cuda and device.type == "cuda":
            torch.cuda.synchronize()
        t0 = perf_counter()

        X = torch.linalg.solve(M.transpose(-2, -1), C.transpose(-2, -1))
        upd_matrix = X.transpose(-2, -1)

        if sync_cuda and device.type == "cuda":
            torch.cuda.synchronize()
        t1 = perf_counter()

    if print_timing:
        print(f"[AlphaEdit formula (solve)] time = {t1 - t0:.6f}s (device: {device})")

    return upd_matrix

def benchmark_all_three(
    d: int = 14336,
    m: int = 2000,
    t: Optional[int] = 1,         # 第 t 次编辑（1..m），默认 m
    r1: int = 1,                     # 当前批次列数
    lam: float = 10.0,               # 与代码里 hparams.L2 对齐
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    sync_cuda: bool = True,
    return_updates: bool = False,
):
    """
    生成:
      P0 = I_d
      K_total ∈ R^{d×m} (N(0,1)/sqrt(d))
      K_p = K_total[:, :t-1], K_1 = K_total[:, t-1:t-1+r1]
      resid ∈ R^{d×r1}
    分别计时三种更新：
      - lowrank_cholesky (更新 P)
      - full_solve      (更新 P)
      - formula_solve   (不更新 P)
    返回 dict，包括三者时间与范数；可选返回三者的更新矩阵。
    """
    from time import perf_counter

    torch.manual_seed(seed)
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    d = int(d); m = int(m)
    if t is None: t = m
    assert 1 <= t <= m, "t 必须在 [1, m] 内"
    assert r1 >= 1 and (t - 1 + r1) <= m, "r1 大小与 t 不可越界"

    # P0 = I_d
    P0 = torch.eye(d, device=dev, dtype=dtype)

    # 总的 K；缩放 1/sqrt(d) 稳定量级
    K_total = torch.randn(d, m, device=dev, dtype=dtype) / (d ** 0.5)

    # 切分 K_p, K_1
    K_p = K_total[:, : (t - 1)] if t > 1 else None
    K_1 = K_total[:, (t - 1) : (t - 1 + r1)]

    # 当前批次残差
    resid = torch.randn(d, r1, device=dev, dtype=dtype) / (d ** 0.5)

    results = {}

    # -------- 1) Low-rank (Cholesky) + 更新 P --------
    P_lr = P0.clone()
    if sync_cuda and dev.type == "cuda": torch.cuda.synchronize()
    t0 = perf_counter()
    with torch.no_grad():
        upd_lr = solve_update_lowrank_cholesky_and_update_P(
            P_i=P_lr, K_t=K_1, resid=resid, lam=float(lam)
        )
        if sync_cuda and dev.type == "cuda": torch.cuda.synchronize()
    t1 = perf_counter()
    time_lr = t1 - t0
    norm_lr = torch.linalg.norm(upd_lr).item()
    results["lowrank_time_s"] = time_lr
    results["lowrank_fro"] = norm_lr

    # -------- 2) Full solve + 更新 P --------
    P_full = P0.clone()
    if sync_cuda and dev.type == "cuda": torch.cuda.synchronize()
    t0 = perf_counter()
    with torch.no_grad():
        upd_full = solve_update_full_and_update_P(
            P_i=P_full, K_t=K_1, resid=resid, lam=float(lam)
        )
        if sync_cuda and dev.type == "cuda": torch.cuda.synchronize()
    t1 = perf_counter()
    time_full = t1 - t0
    norm_full = torch.linalg.norm(upd_full).item()
    results["full_time_s"] = time_full
    results["full_fro"] = norm_full

    # -------- 3) 论文公式（K_p, K_1）solve 版本（不更新 P）--------
    P_form = P0.clone()  # 不会被修改，但保持一致的初值
    if sync_cuda and dev.type == "cuda": torch.cuda.synchronize()
    t0 = perf_counter()
    with torch.no_grad():
        upd_form = alphaedit_formula_update_timed(
            P_i=P_form, resid=resid, K_p=K_p, K_1=K_1,
            sync_cuda=False, print_timing=False
        )
        if sync_cuda and dev.type == "cuda": torch.cuda.synchronize()
    t1 = perf_counter()
    time_form = t1 - t0
    norm_form = torch.linalg.norm(upd_form).item()
    results["formula_time_s"] = time_form
    results["formula_fro"] = norm_form

    # 简要打印
    print(f"[Low-rank]   time={time_lr:.6f}s,  ||Δ||_F={norm_lr:.4e}")
    print(f"[Full-solve] time={time_full:.6f}s, ||Δ||_F={norm_full:.4e}")
    print(f"[Formula]    time={time_form:.6f}s, ||Δ||_F={norm_form:.4e}")

    if return_updates:
        results["upd_lowrank"] = upd_lr
        results["upd_full"] = upd_full
        results["upd_formula"] = upd_form
    else:
        # 释放大矩阵，避免占用显存/内存
        del upd_lr, upd_full, upd_form
        torch.cuda.empty_cache() if dev.type == "cuda" else None

    return results
if __name__ == "__main__":
    import argparse, json, math, sys, time

    def str2dtype(s: str) -> torch.dtype:
        s = s.lower()
        if s in ("fp32","float32"): return torch.float32
        if s in ("fp16","float16"): return torch.float16
        if s in ("bf16","bfloat16"): return torch.bfloat16
        raise ValueError(f"Unsupported dtype: {s}")

    parser = argparse.ArgumentParser("Benchmark AlphaEdit solvers")
    parser.add_argument("--d", type=int, default=14336, help="model hidden size")
    parser.add_argument("--m", type=int, default=10000, help="#total edits (columns of K_total)")
    parser.add_argument("--t", type=str, default="all",
                        help="'all' or integer (1..m) or 'first|mid|last'")
    parser.add_argument("--r1", type=int, default=1, help="#cols in current batch (K_1)")
    parser.add_argument("--lam", type=float, default=10.0, help="L2 (for lowrank/full)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", type=str, default="fp32", help="fp32|fp16|bf16")
    parser.add_argument("--repeats", type=int, default=3, help="#repeats for averaging")
    parser.add_argument("--csv", type=str, default="", help="optional CSV output path")
    args = parser.parse_args()

    dtype = str2dtype(args.dtype)
    dev = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    # 解析 t 集合
    if args.t == "all":
        ts = [1, args.m//2, args.m]
    elif args.t.lower() in ("first","start"):
        ts = [1]
    elif args.t.lower() in ("mid","middle"):
        ts = [args.m//2]
    elif args.t.lower() in ("last","end"):
        ts = [args.m]
    else:
        tval = int(args.t)
        assert 1 <= tval <= args.m
        ts = [tval]

    # 打印表头
    print(f"Device={dev}, DType={dtype}, d={args.d}, m={args.m}, r1={args.r1}, repeats={args.repeats}")
    print(f"{'t':>6} | {'lowrank(s)':>12} {'full(s)':>12} {'formula(s)':>12} || {'||Δ||_lr':>12} {'||Δ||_full':>12} {'||Δ||_form':>12}")
    print("-"*90)

    rows = []
    for t in ts:
        # 预先生成同一组 P0/K_total/resid，保证三法公平
        # 这里调用一次 benchmark_all_three 来初始化随机种子/样本，但重复时需固定相同输入
        # 因此我们在每个重复里手动构造相同样本。
        torch.manual_seed(0)
        P0 = torch.eye(args.d, device=dev, dtype=dtype)
        K_total = torch.randn(args.d, args.m, device=dev, dtype=dtype) / (args.d ** 0.5)
        K_p = K_total[:, : (t-1)] if t > 1 else None
        K_1 = K_total[:, (t-1):(t-1+args.r1)]
        resid = torch.randn(args.d, args.r1, device=dev, dtype=dtype) / (args.d ** 0.5)

        # 计时函数包装
        def time_once():
            # 1) lowrank
            P_lr = P0.clone()
            if dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                upd_lr = solve_update_lowrank_cholesky_and_update_P(P_lr, K_1, resid, float(args.lam))
                if dev.type == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
            tlr = t1 - t0
            nlr = float(torch.linalg.norm(upd_lr))

            # 2) full
            P_full = P0.clone()
            if dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                upd_full = solve_update_full_and_update_P(P_full, K_1, resid, float(args.lam))
                if dev.type == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
            tfull = t1 - t0
            nfull = float(torch.linalg.norm(upd_full))

            # 3) formula
            P_form = P0  # 不会被修改
            if dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                upd_form = alphaedit_formula_update_timed(P_form, resid, K_p=K_p, K_1=K_1,
                                                         sync_cuda=False, print_timing=False)
                if dev.type == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
            tform = t1 - t0
            nform = float(torch.linalg.norm(upd_form))

            # 释放临时更新矩阵，省显存
            del upd_lr, upd_full, upd_form
            if dev.type == "cuda": torch.cuda.empty_cache()
            return tlr, tfull, tform, nlr, nfull, nform

        # 重复多次取平均
        sums = [0.0,0.0,0.0,0.0,0.0,0.0]
        for _ in range(args.repeats):
            tlr, tfull, tform, nlr, nfull, nform = time_once()
            sums = [a+b for a,b in zip(sums, [tlr,tfull,tform,nlr,nfull,nform])]
        avgs = [x/args.repeats for x in sums]

        print(f"{t:6d} | {avgs[0]:12.6f} {avgs[1]:12.6f} {avgs[2]:12.6f} || {avgs[3]:12.4e} {avgs[4]:12.4e} {avgs[5]:12.4e}")
        rows.append({
            "t": t,
            "d": args.d,
            "m": args.m,
            "r1": args.r1,
            "lam": args.lam,
            "device": str(dev),
            "dtype": args.dtype,
            "repeats": args.repeats,
            "lowrank_time_s": avgs[0],
            "full_time_s": avgs[1],
            "formula_time_s": avgs[2],
            "lowrank_fro": avgs[3],
            "full_fro": avgs[4],
            "formula_fro": avgs[5],
        })

    # 可选导出 CSV
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved CSV -> {args.csv}")
