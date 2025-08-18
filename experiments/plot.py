import json
import glob
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 修改为你的目录 ======
RUN_DIR = "/root/autodl-tmp/sic_lyu/AlphaEdit/results/AlphaEdit/run_077"
# ===========================

pattern = os.path.join(RUN_DIR, "forgetting_report_at_*_edits.json")
files = glob.glob(pattern)
assert files, f"No forgetting reports found at: {pattern}"

# 1) 读取所有检查点 -> 三个 dict: {E: {case_id: metric}}
reports_rewrite = {}
reports_paraphrase = {}
reports_neighborhood = {}

for f in files:
    with open(f, "r") as fh:
        data = json.load(fh)
    E = int(data["checkpoint_total_edits"])

    r_dict, p_dict, n_dict = {}, {}, {}
    for item in data.get("detailed_results", []):
        cid = int(item["case_id"])
        # 向后兼容旧报告：没有 rewrite_acc 就用 accuracy
        r_dict[cid] = float(item.get("rewrite_acc", item.get("accuracy", 0.0)))
        p_dict[cid] = float(item.get("paraphrase_acc", 0.0))
        n_dict[cid] = float(item.get("neighborhood_acc", 0.0))

    reports_rewrite[E] = r_dict
    reports_paraphrase[E] = p_dict
    reports_neighborhood[E] = n_dict

# 2) 按检查点排序（以 rewrite 的键为准）
checkpoints = sorted(reports_rewrite.keys())

# 3) 每个检查点的“到目前为止所有 case”的集合（取 rewrite 的 case 集合）
cum_case_sets = OrderedDict()
for E in checkpoints:
    cum_case_sets[E] = set(reports_rewrite[E].keys())

# 4) 定义“分组”：第 k 组 = checkpoint_k 的集合 - checkpoint_{k-1} 的集合
groups = OrderedDict()
prev = set()
for E in checkpoints:
    group_cases = sorted(list(cum_case_sets[E] - prev))
    groups[E] = group_cases
    prev = cum_case_sets[E]

# 5) 对每一组，计算在所有 >= 定义点 的检查点上的平均 accuracy（只用该组固定 ID）
out_dir = os.path.join(RUN_DIR, "forgetting_plots")
os.makedirs(out_dir, exist_ok=True)
all_group_frames = []

for E_def, group_cases in groups.items():
    if not group_cases:
        continue

    xs = []
    ys_r, ys_p, ys_n = [], [], []
    ns_r, ns_p, ns_n = [], [], []

    for E in checkpoints:
        if E < E_def:
            continue

        # 收集该组在此检查点的三类分数
        rw = [reports_rewrite[E][c] for c in group_cases if c in reports_rewrite[E]]
        pp = [reports_paraphrase[E][c] for c in group_cases if c in reports_paraphrase[E]]
        nn = [reports_neighborhood[E][c] for c in group_cases if c in reports_neighborhood[E]]

        if rw or pp or nn:
            xs.append(E)
            ys_r.append(float(np.mean(rw)) if rw else np.nan)
            ys_p.append(float(np.mean(pp)) if pp else np.nan)
            ys_n.append(float(np.mean(nn)) if nn else np.nan)
            ns_r.append(len(rw))
            ns_p.append(len(pp))
            ns_n.append(len(nn))

    if not xs:
        continue

    # 画图（每组一张，三条曲线）
    plt.figure()
    plt.plot(xs, ys_r, label="rewrite")
    plt.plot(xs, ys_p, label="paraphrase")
    plt.plot(xs, ys_n, label="neighborhood")
    gmin, gmax = min(group_cases), max(group_cases)
    plt.title(f"Group defined at {E_def} edits (cases ~{gmin}-{gmax}, size={len(group_cases)})")
    plt.xlabel("Checkpoint total edits")
    plt.ylabel("Average accuracy over group")
    plt.xticks(xs, rotation=45)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"group_{E_def}_edits.png")
    plt.savefig(out_png)
    plt.close()

    # 导出该组的时序 CSV（三列 accuracy + 各自样本数）
    df = pd.DataFrame({
        "group_defined_at_edits":        [E_def]*len(xs),
        "checkpoint_total_edits":        xs,
        "avg_rewrite_accuracy":          ys_r,
        "num_rewrite_cases_present":     ns_r,
        "avg_paraphrase_accuracy":       ys_p,
        "num_paraphrase_cases_present":  ns_p,
        "avg_neighborhood_accuracy":     ys_n,
        "num_neighborhood_cases_present":ns_n,
        "group_case_min_id":             [gmin]*len(xs),
        "group_case_max_id":             [gmax]*len(xs),
        "num_cases_in_group":            [len(group_cases)]*len(xs),
    })
    csv_path = os.path.join(out_dir, f"group_{E_def}_edits.csv")
    df.to_csv(csv_path, index=False)
    all_group_frames.append(df)

# 6) 汇总所有组
if all_group_frames:
    bigdf = pd.concat(all_group_frames, ignore_index=True)
    bigdf.to_csv(os.path.join(out_dir, "all_groups_summary.csv"), index=False)
    print("Done. Plots & CSV saved to:", out_dir)
else:
    print("No groups produced. Check your JSON contents.")
