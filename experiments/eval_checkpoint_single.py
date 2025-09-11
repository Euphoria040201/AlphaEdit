import os, json, time, re
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import (
    AttributeSnippets, get_tfidf_vectorizer,
    CounterFactDataset, MultiCounterFactDataset, MENDQADataset, MQUAKEDataset
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_mquake import compute_rewrite_quality_mquake
from util.globals import DATA_DIR, RESULTS_DIR

DS_DICT = {
    "cf":   (CounterFactDataset,       compute_rewrite_quality_counterfact),
    "mcf":  (MultiCounterFactDataset,  compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset,            compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset,          compute_rewrite_quality_mquake),
}

def load_model(local_ckpt: str, trust_remote_code: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"[Eval] Loading model from: {local_ckpt}")
    model = AutoModelForCausalLM.from_pretrained(
        local_ckpt,
        torch_dtype=torch.float32,
        trust_remote_code=trust_remote_code
    ).cuda()
    tok = AutoTokenizer.from_pretrained(local_ckpt, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok

def _infer_n_from_ckpt(ckpt_dir: str) -> Optional[int]:
    # 1) meta.json
    meta = Path(ckpt_dir) / "meta.json"
    if meta.exists():
        try:
            with open(meta, "r") as f:
                j = json.load(f)
            n = int(j.get("total_edits", 0))
            if n > 0:
                return n
        except Exception:
            pass
    # 2) 目录名 edits_000100
    m = re.search(r"edits_(\d+)", Path(ckpt_dir).name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None

def _find_run_root(ckpt_dir: Path) -> Optional[Path]:
    """
    从 ckpt 路径向上找最近的 run_XXX 目录（例如 run_090）。
    找不到就返回 None。
    """
    for p in ckpt_dir.parents:
        if re.match(r"run_\d+$", p.name):
            return p
    return None

def main(
    ckpt_dir: str,
    ds_name: str = "mcf",
    dataset_size_limit: int = None,
    generation_test_interval: int = 1,
    out_run_name: str = "eval_only",
    trust_remote_code: bool = False,
    eval_before_n_edits: Optional[int] = None,   # 只评测前 N 条（按数据集顺序）
    eval_case_id_max: Optional[int] = None,      # 只评测 case_id <= K
    infer_from_ckpt_meta: bool = False,          # 自动从 ckpt 推断 N
    to_run_dir: bool = True,                     # ★ 写到对应 run 目录
    overwrite: bool = False,                     # ★ 允许覆盖已有 case_*.json
):
    ckpt_dir = Path(ckpt_dir)
    model, tok = load_model(str(ckpt_dir), trust_remote_code=trust_remote_code)

    ds_class, ds_eval = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    print(f"[Eval] Loaded dataset {ds_name} with {len(ds)} records")

    # 只在需要生成评测时加载
    snips = AttributeSnippets(DATA_DIR) if generation_test_interval != -1 else None
    vec   = get_tfidf_vectorizer(DATA_DIR) if generation_test_interval != -1 else None

    # 推断 N（若未显式给出且开启了自动推断）
    if eval_before_n_edits is None and infer_from_ckpt_meta:
        inferred = _infer_n_from_ckpt(str(ckpt_dir))
        if inferred is not None:
            eval_before_n_edits = inferred
            print(f"[Eval] infer_from_ckpt_meta => only first {eval_before_n_edits} items will be evaluated.")

    # ===== 输出目录放到对应 run 目录下 =====
    if to_run_dir:
        run_root = _find_run_root(ckpt_dir)
        if run_root is not None:
            # e.g. results/AlphaEdit/run_090/eval/edits_000100/mcf/
            out_dir = run_root / "eval" / ckpt_dir.name / ds_name
        else:
            # 找不到 run_XXX，就退回到传统 RESULTS_DIR/out_run_name/ds_name
            print("[Eval][WARN] cannot locate run_XXX from ckpt; fallback to RESULTS_DIR/out_run_name/ds_name")
            out_dir = RESULTS_DIR / out_run_name / ds_name
    else:
        out_dir = RESULTS_DIR / out_run_name / ds_name

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Eval] Results will be saved under: {out_dir}")

    # 记录一下评测过滤条件
    with open(out_dir / "filters.json", "w") as f:
        json.dump({
            "eval_before_n_edits": eval_before_n_edits,
            "eval_case_id_max": eval_case_id_max,
            "infer_from_ckpt_meta": infer_from_ckpt_meta,
            "ckpt_dir": str(ckpt_dir),
            "ds_name": ds_name,
            "dataset_size_limit": dataset_size_limit,
            "generation_test_interval": generation_test_interval,
        }, f, indent=2)

    t0 = time.time()
    summary = []
    for i, record in enumerate(ds):
        # 过滤：只评测“之前的”
        if eval_before_n_edits is not None and i >= eval_before_n_edits:
            continue
        if eval_case_id_max is not None and int(record["case_id"]) > eval_case_id_max:
            continue

        out_file = out_dir / f"case_{record['case_id']}.json"
        if out_file.exists() and not overwrite:
            continue

        do_gen = (generation_test_interval != -1 and (record["case_id"] % generation_test_interval == 0))
        m = ds_eval(
            model, tok, record,
            *( [snips, vec] if do_gen else [None, None] )
        )
        obj = {
            "case_id": record["case_id"],
            "requested_rewrite": record["requested_rewrite"],
            "post": m,
        }
        with open(out_file, "w") as f:
            json.dump(obj, f, indent=2)
        summary.append(obj)

        if (len(summary)) % 50 == 0:
            print(f"[Eval] {len(summary)} evaluated ...")

    # ===== 生成 summary.json =====
    def _mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    rewrite_acc, para_acc, neigh_acc = [], [], []
    forgetting_results = []  # 用于 forgetting_report.json 的详细条目

    for obj in summary:
        post = obj.get("post", {})
        # 三项在项目里是 list[bool]，取均值即可
        rw = np.mean(post.get("rewrite_prompts_correct", [])) if "rewrite_prompts_correct" in post else None
        pp = np.mean(post.get("paraphrase_prompts_correct", [])) if "paraphrase_prompts_correct" in post else None
        ng = np.mean(post.get("neighborhood_prompts_correct", [])) if "neighborhood_prompts_correct" in post else None

        if rw is not None: rewrite_acc.append(rw)
        if pp is not None: para_acc.append(pp)
        if ng is not None: neigh_acc.append(ng)

        forgetting_results.append({
            "case_id": int(obj["case_id"]),
            "rewrite_acc": float(rw) if rw is not None else None,
            "paraphrase_acc": float(pp) if pp is not None else None,
            "neighborhood_acc": float(ng) if ng is not None else None
        })

    sum_obj = {
        "ds": ds_name,
        "n_cases": len(summary),
        "rewrite_acc_mean": _mean(rewrite_acc),
        "paraphrase_acc_mean": _mean(para_acc),
        "neighborhood_acc_mean": _mean(neigh_acc),
        "time_sec": float(time.time() - t0),
        "ckpt_dir": str(ckpt_dir),
        "out_dir": str(out_dir),
        "filters": {
            "eval_before_n_edits": eval_before_n_edits,
            "eval_case_id_max": eval_case_id_max,
            "infer_from_ckpt_meta": infer_from_ckpt_meta,
        }
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(sum_obj, f, indent=2)
    print("[Eval][SUMMARY]", json.dumps(sum_obj, indent=2))

    # ===== 生成 forgetting_report.json（同你之前结构风格）=====
    # checkpoint_total_edits：优先取 meta.json 的 total_edits；否则用目录名里的数字；再否则用 n_cases
    n_from_meta = _infer_n_from_ckpt(str(ckpt_dir))
    checkpoint_total_edits = int(n_from_meta if n_from_meta is not None else len(summary))

    fr_obj = {
        "checkpoint_total_edits": checkpoint_total_edits,
        "num_edits_tested": len(summary),
        "average_rewrite_accuracy":  f"{_mean(rewrite_acc):.4f}",
        "average_paraphrase_accuracy": f"{_mean(para_acc):.4f}",
        "average_neighborhood_accuracy": f"{_mean(neigh_acc):.4f}",
        "detailed_results": forgetting_results
    }
    with open(out_dir / "forgetting_report.json", "w") as f:
        json.dump(fr_obj, f, indent=2)
    print(f"[Eval] Forgetting report saved to {out_dir / 'forgetting_report.json'}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--ds_name", choices=list(DS_DICT.keys()), default="mcf")
    ap.add_argument("--dataset_size_limit", type=int, default=None)
    ap.add_argument("--generation_test_interval", type=int, default=1, help="-1 只做快评测；>=1 每 N 条做一次生成评测")
    ap.add_argument("--out_run_name", default="eval_only")
    ap.add_argument("--trust_remote_code", action="store_true")
    # 过滤参数
    ap.add_argument("--eval_before_n_edits", type=int, default=None)
    ap.add_argument("--eval_case_id_max", type=int, default=None)
    ap.add_argument("--infer_from_ckpt_meta", action="store_true")
    # 行为控制
    ap.add_argument("--to_run_dir", action="store_true", help="把输出落到该 checkpoint 所在 run 目录（推荐）")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()
    main(
        ckpt_dir=args.ckpt_dir,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        generation_test_interval=args.generation_test_interval,
        out_run_name=args.out_run_name,
        trust_remote_code=args.trust_remote_code,
        eval_before_n_edits=args.eval_before_n_edits,
        eval_case_id_max=args.eval_case_id_max,
        infer_from_ckpt_meta=args.infer_from_ckpt_meta,
        to_run_dir=args.to_run_dir,
        overwrite=args.overwrite,
    )
