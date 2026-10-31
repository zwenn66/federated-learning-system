# -*- coding: utf-8 -*-
"""离屏验证：_parse_progress 会话边界——第二次训练(ROUND 重新从1开始)时丢弃上一会话指标"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import federated_gui_pyside as m

app = m.QApplication([])
win = m.FederatedGUI()

# 构造模拟日志：第一次训练 14 轮 + 第二次训练 3 轮（ROUND 重新从 1 开始）
log = os.path.join(PROJ, "_parse_test.log")
with open(log, "w", encoding="utf-8") as f:
    for r in range(1, 15):   # 第一次会话 14 轮
        f.write(f"[ROUND {r}] FitMetrics(train_loss: {1.5-r*0.1:.2f}, train_acc: {0.4+r*0.03:.4f})\n")
        f.write(f"[ROUND {r}] EvaluateMetrics(test_acc: {0.3+r*0.02:.4f})\n")
    for r in range(1, 4):    # 第二次会话 3 轮（ROUND 重新从 1 开始）
        f.write(f"[ROUND {r}] FitMetrics(train_loss: {2.0-r*0.2:.2f}, train_acc: {0.1+r*0.05:.4f})\n")
        f.write(f"[ROUND {r}] EvaluateMetrics(test_acc: {0.1+r*0.04:.4f})\n")

# 用测试日志覆盖 _server_log_path
win._server_log_path = lambda: log
cur, metrics, new_session = win._parse_progress()
print("[1] 解析后当前轮:", cur, "(应为 3)")
print("[2] 解析出轮次:", sorted(metrics.keys()), "(应只含 1,2,3 = 第二次会话)")
print("[3] 第1轮 test_acc:", metrics.get(1, {}).get("test_acc"), "(应为 0.14=新会话, 而非 0.32=旧会话)")
assert cur == 3
assert sorted(metrics.keys()) == [1, 2, 3]
assert metrics[1]["test_acc"] > 0.13 and metrics[1]["test_acc"] < 0.15
assert new_session is True, "应识别到会话回退"

# 单会话日志（第一次启动，无历史）：不应误清空
with open(log, "w", encoding="utf-8") as f:
    for r in range(1, 5):
        f.write(f"[ROUND {r}] FitMetrics(train_loss: 1.0, train_acc: 0.5)\n")
        f.write(f"[ROUND {r}] EvaluateMetrics(test_acc: 0.4)\n")
cur, metrics, new_session = win._parse_progress()
print("[4] 单会话解析: cur =", cur, "轮次 =", sorted(metrics.keys()), "(应 1..4 全保留)")
assert cur == 4 and sorted(metrics.keys()) == [1, 2, 3, 4]
assert new_session is False, "单会话不应触发会话回退"

os.remove(log)
print("== 会话边界验证完成 ==")
