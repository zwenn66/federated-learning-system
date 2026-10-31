# -*- coding: utf-8 -*-
"""安全对抗实验：缩放攻击 vs 鲁棒聚合防御（Trimmed Mean / Krum）

实验矩阵（5 客户端、5 轮、Non-IID 每客户端 2 类、缩放攻击 ×50）：
  baseline : 0 个恶意 · FedAvg（defense=none）
  attack1  : 1 个恶意（client_1）· FedAvg
  defend1  : 1 个恶意（client_1）· Trimmed Mean（trim_fraction=0.2）
  attack2  : 2 个恶意（client_1/client_2）· FedAvg
  defend2  : 2 个恶意（client_1/client_2）· Trimmed Mean（trim_fraction=0.2）

主断言（防御有效性，恶意 20% 场景）：
  1. 攻击显著拉低准确率（attack1 < baseline - 0.08）
  2. 防御显著恢复（defend1 > attack1 + 0.10）
  3. 防御效用代价可控（defend1 >= baseline - 0.20）
辅助断言（攻击威胁，恶意 40% 场景）：
  4. attack2 < baseline - 0.08

防御边界（安全讨论，不断言）：
  defend2 组恶意占 40%，Trimmed Mean 单参数裁剪（每端 20% 只裁 1 个）不足以
  剔除全部恶意更新——恶意占比接近 50% 时"多数诚实"假设失效，任何鲁棒聚合
  都受限（Krum 需 n ≥ 2f+3）。这是真实的安全局限，README 中作为边界讨论。

脚本开头先做算法级自检（纯函数，秒级）：构造 5 客户端权重 + 1 个放大 50 倍
的恶意权重，验证 trimmed_mean / krum 能恢复接近诚实均值，而 FedAvg 被带偏。

运行前会临时改写 server_config_gui.json / 清理 server.log / client_logs，结束后自动还原现场。
用法：.\\.venv\\Scripts\\python scripts\\verify_poison_defense.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)  # 复用服务器中的鲁棒聚合实现做自检

from federated_server import trimmed_mean_aggregate, krum_aggregate  # noqa: E402

PY = sys.executable
SERVER_CFG = os.path.join(PROJ, "server_config_gui.json")
GUI_CFG = os.path.join(PROJ, "gui_config.json")
LOG = os.path.join(PROJ, "server.log")
CLIENT_LOGS = os.path.join(PROJ, "client_logs")
MODELS = os.path.join(PROJ, "saved_models")

NUM_CLIENTS = 5
NUM_ROUNDS = 5
SCALE = 50.0
PORTS = {"baseline": 8083, "attack1": 8086, "defend1": 8089,
         "attack2": 8092, "defend2": 8095}


# ---------- 算法级自检（不跑训练，直接验证聚合函数行为） ----------
def algo_sanity_check():
    """构造 5 组权重，client_4 为放大 50 倍的恶意更新（恶意占比 20%，
    该场景下 Trimmed Mean k=1 / Krum f=1 理论有效），验证 trimmed_mean / krum
    能恢复接近诚实均值，而普通平均被带偏。恶意 40% 场景属于防御边界，
    由完整实验 defend2 组展示，不做"恢复"断言。"""
    rng = np.random.default_rng(0)
    n_layers = 3
    honest = []
    for _ in range(4):
        honest.append([rng.normal(0, 0.1, size=(4, 4)).astype(np.float32),
                       rng.normal(0, 0.01, size=(4,)).astype(np.float32),
                       rng.normal(0, 0.1, size=(4, 4)).astype(np.float32)])
    malicious = [[np.asarray(w) * SCALE for w in h] for h in honest[:1]]

    fedavg = [np.mean([w[i] for w in honest + malicious], axis=0) for i in range(n_layers)]
    tm = trimmed_mean_aggregate(honest + malicious, trim_fraction=0.2)
    kr = krum_aggregate(honest + malicious, num_malicious=1)

    honest_mean = [np.mean([w[i] for w in honest], axis=0) for i in range(n_layers)]

    def err(a, b):
        return max(float(np.max(np.abs(np.asarray(ai, dtype=np.float32)
                                       - np.asarray(bi, dtype=np.float32))))
                   for ai, bi in zip(a, b))
    avg_err = err(fedavg, honest_mean)
    tm_err = err(tm, honest_mean)
    kr_err = err(kr, honest_mean)
    tm_ok = tm_err < avg_err * 0.2
    kr_ok = kr_err < avg_err * 0.2
    avg_bad = avg_err > 1.0
    print(f"  [自检] 普通平均被恶意带偏 (err={avg_err:.4f} > 1.0): "
          + ("PASS" if avg_bad else "FAIL"))
    print(f"  [自检] Trimmed Mean 缩小带偏 (err={tm_err:.4f} < {avg_err * 0.2:.4f}): "
          + ("PASS" if tm_ok else "FAIL"))
    print(f"  [自检] Krum 缩小带偏 (err={kr_err:.4f} < {avg_err * 0.2:.4f}): "
          + ("PASS" if kr_ok else "FAIL"))
    return tm_ok and kr_ok and avg_bad


# ---------- 实验编排 ----------
def parse_curve(log_text):
    """解析 server.log 中每轮 EvaluateMetrics 的 test_acc → {round: acc}"""
    curve = {}
    for m in re.finditer(r"\[ROUND (\d+)\] EvaluateMetrics = \{(.*?)\}", log_text, re.S):
        r = int(m.group(1))
        acc_m = re.search(r"test_acc['\"]?\s*[:=]\s*([0-9.]+)", m.group(2))
        if acc_m:
            curve[r] = float(acc_m.group(1))
    return curve


def wait_for(pred, timeout=600, interval=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def run_experiment(name, malicious_cids, defense="none"):
    """运行一组实验，返回 (是否完成, 每轮test_acc曲线)"""
    port = PORTS[name]
    cfg = {
        "port": port, "num_rounds": NUM_ROUNDS,
        "min_fit_clients": NUM_CLIENTS, "min_evaluate_clients": NUM_CLIENTS,
        "min_available_clients": NUM_CLIENTS,
        "fraction_fit": 1.0, "fraction_evaluate": 1.0,
        "local_epochs": 1, "model_save_path": "./saved_models",
        "save_interval": 999,  # 本实验不关注模型落盘，避免生成模型文件
        "defense": defense,
        "trim_fraction": 0.2,
        "num_malicious": 1 if len(malicious_cids) <= 1 else 2,
    }
    with open(SERVER_CFG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    if os.path.exists(LOG):
        os.remove(LOG)

    procs = []
    try:
        p = subprocess.Popen([PY, "federated_server.py", "--config", SERVER_CFG],
                             cwd=PROJ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        procs.append(p)
        time.sleep(4)
        for i in range(1, NUM_CLIENTS + 1):
            cid = f"client_{i}"
            cmd = [PY, "federated_client.py", "--cid", cid,
                   "--server", f"127.0.0.1:{port}", "--num-clients", str(NUM_CLIENTS)]
            if cid in malicious_cids:
                cmd += ["--malicious", "scale", "--scale", str(SCALE)]
            c = subprocess.Popen(cmd, cwd=PROJ,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
            procs.append(c)
            time.sleep(1.5)

        target = f"[ROUND {NUM_ROUNDS}] EvaluateMetrics"
        done = wait_for(lambda: os.path.exists(LOG) and target in
                        open(LOG, "r", encoding="utf-8", errors="ignore").read())
        curve = {}
        if os.path.exists(LOG):
            curve = parse_curve(open(LOG, "r", encoding="utf-8", errors="ignore").read())
        return done, curve
    finally:
        for proc in procs:
            try:
                proc.kill()
            except Exception:
                pass
        time.sleep(2)


EXPERIMENTS = [
    ("baseline", set(), "none"),
    ("attack1", {"client_1"}, "none"),
    ("defend1", {"client_1"}, "trimmed_mean"),
    ("attack2", {"client_1", "client_2"}, "none"),
    ("defend2", {"client_1", "client_2"}, "trimmed_mean"),
]


def main():
    print(f"==== 安全对抗实验：缩放攻击(×{SCALE:g}) vs Trimmed Mean 防御 ====")
    print(f"客户端 {NUM_CLIENTS} 个 · 轮数 {NUM_ROUNDS} · 恶意占比 0 / 20% / 40%")
    print("=" * 60)

    print("---- 算法级自检（纯函数） ----")
    if not algo_sanity_check():
        print("❌ 算法自检失败，终止实验（请检查 federated_server.py 的聚合实现）")
        sys.exit(1)

    # ---------- 备份现场 ----------
    backup = {}
    for p in (SERVER_CFG, GUI_CFG):
        if os.path.exists(p):
            backup[p] = open(p, "r", encoding="utf-8").read()
    log_backup = None
    if os.path.exists(LOG):
        log_backup = open(LOG, "r", encoding="utf-8").read()
    model_before = set(os.listdir(MODELS)) if os.path.isdir(MODELS) else set()
    if os.path.isdir(CLIENT_LOGS):
        shutil.rmtree(CLIENT_LOGS)

    results = {}
    try:
        for name, cids, defense in EXPERIMENTS:
            done, curve = run_experiment(name, cids, defense)
            results[name] = curve
            print(f"[完成] {name:<9} 恶意={len(cids)} 防御={defense or 'none'} 完成={done} 末轮={curve.get(NUM_ROUNDS) if curve else '—'}")
    finally:
        # ---------- 还原现场 ----------
        for key, val in backup.items():
            with open(key, "w", encoding="utf-8") as f:
                f.write(val)
        if log_backup is not None:
            with open(LOG, "w", encoding="utf-8") as f:
                f.write(log_backup)
        elif os.path.exists(LOG):
            os.remove(LOG)
        if os.path.isdir(MODELS):
            now = set(os.listdir(MODELS))
            for f in now - model_before:
                try:
                    os.remove(os.path.join(MODELS, f))
                except Exception:
                    pass
        if os.path.isdir(CLIENT_LOGS):
            shutil.rmtree(CLIENT_LOGS)
        print("== 现场已还原（配置/日志/模型/客户端日志）==")

    # ---------- 输出与断言 ----------
    print("\n==== 每轮 test_acc 对比 ====")
    header = f"{'轮次':<6}{'基线':<10}{'攻击20%':<10}{'防御20%':<10}{'攻击40%':<10}{'防御40%':<10}"
    print(header)
    all_rounds = sorted(set().union(*(c.keys() for c in results.values())))
    for r in all_rounds:
        row = [f"R{r}"]
        for name, _, _ in EXPERIMENTS:
            acc = results[name].get(r)
            row.append(f"{acc:.4f}" if acc is not None else "—")
        print("  ".join(f"{x:<10}" for x in row))

    def final_acc(name):
        return results[name].get(NUM_ROUNDS) if results[name] else None

    base_f = final_acc("baseline")
    a1_f = final_acc("attack1")
    d1_f = final_acc("defend1")
    a2_f = final_acc("attack2")
    d2_f = final_acc("defend2")
    print(f"\n末轮 test_acc：基线={base_f}  攻击20%={a1_f}  防御20%={d1_f}"
          f"  攻击40%={a2_f}  防御40%={d2_f}")

    checks = []

    def check(name, cond, detail):
        checks.append((name, cond, detail))

    if base_f is not None and a1_f is not None:
        check("攻击(20%)显著拉低准确率", a1_f < base_f - 0.08,
              f"attack1 {a1_f:.4f} vs baseline {base_f:.4f}")
    else:
        check("攻击(20%)显著拉低准确率", False, "末轮数据缺失")
    if a1_f is not None and d1_f is not None:
        check("防御(20%)显著恢复准确率", d1_f > a1_f + 0.10,
              f"defend1 {d1_f:.4f} vs attack1 {a1_f:.4f}")
    else:
        check("防御(20%)显著恢复准确率", False, "末轮数据缺失")
    if base_f is not None and d1_f is not None:
        check("防御(20%)效用代价可控", d1_f >= base_f - 0.20,
              f"defend1 {d1_f:.4f} vs baseline {base_f:.4f}")
    else:
        check("防御(20%)效用代价可控", False, "末轮数据缺失")
    if base_f is not None and a2_f is not None:
        check("攻击(40%)显著拉低准确率", a2_f < base_f - 0.08,
              f"attack2 {a2_f:.4f} vs baseline {base_f:.4f}")
    else:
        check("攻击(40%)显著拉低准确率", False, "末轮数据缺失")

    print("\n==== 断言结果 ====")
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        all_ok = all_ok and ok

    print("\n==== 防御边界说明（恶意 40% 场景） ====")
    if d2_f is not None and a2_f is not None:
        diff = d2_f - a2_f
        print(f"  防御40%末轮 {d2_f:.4f} vs 攻击40% {a2_f:.4f}（差 {diff:+.4f}）")
    print("  恶意占比接近 50% 时'多数诚实'假设失效，Trimmed Mean 单参数裁剪不足以"
          "\n  剔除全部恶意更新（Krum 需 n ≥ 2f+3）。此为真实安全局限，"
          "\n  提高客户端总数或增大裁剪比例可缓解——详见 README 安全讨论。")

    print(f"\n==== 安全对抗实验：{'✅ 全部通过' if all_ok else '❌ 存在失败'} ====")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
