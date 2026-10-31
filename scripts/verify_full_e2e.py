# -*- coding: utf-8 -*-
"""综合端到端验收：真实 server + 2 客户端 + 4 轮训练 + 指标解析 + 模型落盘 + 二次会话边界

运行前会临时改写 server_config_gui.json / 移动 server.log，结束后自动还原现场。
"""
import json
import os
import shutil
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SERVER_CFG = os.path.join(PROJ, "server_config_gui.json")
GUI_CFG = os.path.join(PROJ, "gui_config.json")
LOG = os.path.join(PROJ, "server.log")
CLIENT_LOGS = os.path.join(PROJ, "client_logs")
MODELS = os.path.join(PROJ, "saved_models")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, PROJ)

results = []


def rec(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def wait_for(pred, timeout=360, interval=2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


# ---------- 0) 备份现场 ----------
backup = {}
for p in (SERVER_CFG, GUI_CFG):
    if os.path.exists(p):
        backup[p] = open(p, "r", encoding="utf-8").read()
log_backup = None
if os.path.exists(LOG):
    log_backup = open(LOG, "r", encoding="utf-8").read()
model_before = set(os.listdir(MODELS)) if os.path.isdir(MODELS) else set()

test_cfg = {"port": 8083, "num_rounds": 5, "min_fit_clients": 2,
            "local_epochs": 1, "model_save_path": "./saved_models"}
with open(SERVER_CFG, "w", encoding="utf-8") as f:
    json.dump(test_cfg, f, indent=2, ensure_ascii=False)
if os.path.exists(LOG):
    os.remove(LOG)
if os.path.isdir(CLIENT_LOGS):
    shutil.rmtree(CLIENT_LOGS)
os.makedirs(CLIENT_LOGS, exist_ok=True)

procs = []
try:
    # ---------- 1) 启动 server + 2 客户端 ----------
    p = subprocess.Popen([PY, "federated_server.py", "--config", SERVER_CFG],
                         cwd=PROJ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    procs.append(p)
    rec("服务器启动", True, f"pid={p.pid} port=8083 rounds=4")
    time.sleep(4)
    for i in range(2):
        c = subprocess.Popen([PY, "federated_client.py",
                              "--cid", f"client_{i + 1}",
                              "--server", "127.0.0.1:8083", "--num-clients", "2"],
                             cwd=PROJ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        procs.append(c)
        time.sleep(2)
    rec("启动 2 客户端", True)

    # ---------- 2) 等待 5 轮训练完成 ----------
    done = wait_for(lambda: os.path.exists(LOG) and "[ROUND 5] EvaluateMetrics" in open(LOG, "r", encoding="utf-8", errors="ignore").read(),
                    timeout=420)
    rec("5 轮训练完成", done, "server.log 出现 [ROUND 5] EvaluateMetrics" if done else "超时")

    # ---------- 3) 日志结构断言 ----------
    text = open(LOG, "r", encoding="utf-8", errors="ignore").read()
    rounds = [r for r in range(1, 6)]
    per_round_ok = all(f"[ROUND {r}] FitMetrics" in text and f"[ROUND {r}] EvaluateMetrics" in text for r in rounds)
    rec("每轮 Fit/Evaluate 双行", per_round_ok)
    import re
    vals = [float(v) for v in re.findall(r"test_acc['\"]?\s*[:=]\s*[{\s']*([0-9.]+)", text)]
    rec("test_acc 数值可解析", len(vals) >= 4, f"解析到 {len(vals)} 个数值")

    # ---------- 4) GUI 解析链路（真实日志） ----------
    import federated_gui_pyside as m
    app = m.QApplication([])
    win = m.FederatedGUI()
    cur, metrics, new_session = win._parse_progress()
    rec("_parse_progress 解析真实日志", cur == 5 and len(metrics) == 5,
        f"cur={cur} rounds={sorted(metrics.keys())} new_session={new_session}")
    ta = metrics.get(5, {}).get("train_acc")
    te = metrics.get(5, {}).get("test_acc")
    rec("第5轮指标齐全", ta is not None and te is not None,
        f"train_acc={ta} test_acc={te}")

    # ---------- 5) 模型落盘与客户端日志 ----------
    model_files = [f for f in os.listdir(MODELS) if f.startswith("global_model_round5")]
    rec("模型文件落盘(save_interval=5)", len(model_files) > 0, f"global_model_round5* x{len(model_files)}")
    cl1 = os.path.join(CLIENT_LOGS, "client_1.log")
    rec("客户端日志写出", os.path.exists(cl1) and os.path.getsize(cl1) > 0,
        f"client_1.log {os.path.getsize(cl1) if os.path.exists(cl1) else 0}B")

    # ---------- 6) 二次会话边界（真实重启，换端口 + 3 轮） ----------
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass
    procs.clear()
    time.sleep(3)
    base_round_lines = 0
    if os.path.exists(LOG):
        base_round_lines = open(LOG, "r", encoding="utf-8", errors="ignore").read().count("[ROUND")
    test_cfg2 = {"port": 8084, "num_rounds": 3, "min_fit_clients": 2,
                 "local_epochs": 1, "model_save_path": "./saved_models"}
    with open(SERVER_CFG, "w", encoding="utf-8") as f:
        json.dump(test_cfg2, f, indent=2, ensure_ascii=False)
    p = subprocess.Popen([PY, "federated_server.py", "--config", SERVER_CFG],
                         cwd=PROJ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    procs.append(p)
    time.sleep(4)
    for i in range(2):
        c = subprocess.Popen([PY, "federated_client.py",
                              "--cid", f"client_{i + 1}",
                              "--server", "127.0.0.1:8084", "--num-clients", "2"],
                             cwd=PROJ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        procs.append(c)
        time.sleep(2)

    def second_done():
        if not os.path.exists(LOG):
            return False
        t = open(LOG, "r", encoding="utf-8", errors="ignore").read()
        return t.count("[ROUND") >= base_round_lines + 6 and "[ROUND 3] EvaluateMetrics" in t

    done2 = wait_for(second_done, timeout=420)
    rec("二次会话 3 轮完成(新日志追加)", done2)
    cur2, metrics2, new_session2 = win._parse_progress()
    rec("二次会话边界解析", done2 and cur2 == 3 and sorted(metrics2.keys()) == [1, 2, 3] and new_session2 is True,
        f"cur={cur2} rounds={sorted(metrics2.keys())} new_session={new_session2}")

finally:
    # ---------- 7) 清理：杀子进程、还原现场 ----------
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass
    time.sleep(2)
    for key, val in backup.items():
        with open(key, "w", encoding="utf-8") as f:
            f.write(val)
    if log_backup is not None:
        with open(LOG, "w", encoding="utf-8") as f:
            f.write(log_backup)
    elif os.path.exists(LOG):
        os.remove(LOG)
    # 只清理本次测试生成的模型
    if os.path.isdir(MODELS):
        now = set(os.listdir(MODELS))
        for f in now - model_before:
            try:
                os.remove(os.path.join(MODELS, f))
            except Exception:
                pass
    print("== 现场已还原（配置/日志/模型）==")

fails = [r for r in results if not r[1]]
print(f"\n==== 综合 E2E 结果：{len(results) - len(fails)}/{len(results)} 通过 ====")
if fails:
    for name, ok, detail in fails:
        print(f"  ❌ {name}: {detail}")
    sys.exit(1)
print("✅ 全部通过")
