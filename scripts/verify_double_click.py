# -*- coding: utf-8 -*-
"""离屏验证：
1) 10 页布局改造后可正常构建（居中限宽容器）
2) 二次训练：旧日志残留时点开始训练不得误判"已完成100%"，新会话 ROUND 1 出现后才正常
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import federated_gui_pyside as m

app = m.QApplication([])
win = m.FederatedGUI()
print("[0] GUI 实例化 OK，10 页构建无异常")

# 构造模拟：旧会话日志（15 轮已完成的上一轮训练）
log = os.path.join(PROJ, "_double_click_test.log")
with open(log, "w", encoding="utf-8") as f:
    for r in range(1, 16):
        f.write(f"[ROUND {r}] FitMetrics(train_loss: {1.0:.2f}, train_acc: {0.9:.4f})\n")
        f.write(f"[ROUND {r}] EvaluateMetrics(test_acc: {0.8:.4f})\n")
win._server_log_path = lambda: log

# 模拟 start_training：total=15（用户配置 15 轮，旧日志也是 15 轮 → 之前必误判完成）
win.total_rounds = 15
win.is_training = True
win.current_round = 0
win._metrics = {}
win._last_plotted_round = -1
win._conn_history = []
win._session_had_history = win._peek_round() >= 1
win._session_new_round_seen = False
win._session_click_round = win._peek_round()   # 模拟 start_training 的点击基准
print(f"[1] start_training: 检测到旧会话 history={win._session_had_history} (应 True)")

# 模拟 _refresh_training（新训练尚未开始，日志还是旧的）
win._refresh_training()
r1 = (win.current_round, win.progress.value(), win.train_status.text())
print(f"[2] 第一次点击后(新训练未开始): 轮次={r1[0]} 进度={r1[1]}% 状态={r1[2]}")
print(f"    → 不误判完成 = {r1[0] == 0 and r1[1] <= 0 and '已完成' not in r1[2]}")
assert r1[0] == 0 and r1[1] <= 0 and "已完成" not in r1[2], "❌ 旧日志残留误判完成!"

# 新训练开始：ROUND 1..3 追加（第二次测试真实 3 轮）
with open(log, "a", encoding="utf-8") as f:
    for r in range(1, 4):
        f.write(f"[ROUND {r}] FitMetrics(train_loss: {2.0-r*0.2:.2f}, train_acc: {0.1+r*0.05:.4f})\n")
        f.write(f"[ROUND {r}] EvaluateMetrics(test_acc: {0.1+r*0.04:.4f})\n")
win._refresh_training()
r2 = (win.current_round, win.progress.value(), win.train_status.text())
print(f"[3] 新会话开始后: 轮次={r2[0]}/15 进度={r2[1]}% 状态={r2[2]}")
assert r2[0] == 3 and r2[1] == 20, "❌ 新会话轮次解析错误"
print("    → 新会话正常推进 ✓")

os.remove(log)
print("== 二次训练防误判验证完成 ==")
