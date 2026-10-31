# -*- coding: utf-8 -*-
"""验证训练流程：启动服务器/客户端后进度条不得自动变化，点开始训练后才动"""
import json
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

# 3 轮快速配置
with open(os.path.join(PROJ, "gui_config.json"), "w", encoding="utf-8") as f:
    json.dump({"server_port": 8099, "num_rounds": 3, "min_clients": 2,
               "local_epochs": 1, "model_save_path": "./saved_models"}, f)
with open(os.path.join(PROJ, "server_config_gui.json"), "w", encoding="utf-8") as f:
    json.dump({"port": 8099, "num_rounds": 3, "min_fit_clients": 2,
               "local_epochs": 1, "model_save_path": "./saved_models"}, f)
log = os.path.join(PROJ, "server.log")
if os.path.exists(log):
    os.remove(log)

import federated_gui_pyside as m
app = m.QApplication([])
win = m.FederatedGUI()

ok = True

# 1) 只启动服务器：进度/训练状态不得自动激活
win.start_server()
time.sleep(3)
r1 = (win.is_training is False, win.progress.value() == 0)
print(f"[1] 启动服务器后  is_training={win.is_training} (应 False)  进度={win.progress.value()}% (应 0)")
if not all(r1):
    ok = False

# 2) 启动客户端：训练在后台自动开始（Flower 行为），但 GUI 监控不得自动激活
win.client_count_spin.setValue(2)
win.start_clients()
time.sleep(12)
print(f"[2] 启动客户端后  is_training={win.is_training} (应 False)  进度={win.progress.value()}% (应 0)  轮次={win.current_round}")
if win.is_training or win.progress.value() != 0:
    ok = False

# 3) 点开始训练：监控激活，进度随真实训练推进（离屏无定时器，手动模拟刷新）
win.start_training()
win._refresh_training()
time.sleep(3)
act1 = (win.is_training, win.current_round, win.progress.value())
print(f"[3] 点开始训练后  is_training={act1[0]}  轮次={act1[1]}  进度={act1[2]}%")
for _ in range(10):     # 模拟 2 秒定时器刷新
    win._refresh_training()
    time.sleep(1.2)
act2 = (win.is_training, win.current_round, win.progress.value())
print(f"[4] 刷新后         is_training={act2[0]}  轮次={act2[1]}  进度={act2[2]}%")
# 3 轮快速训练可能在刷新窗口内完成 → is_training 自动变 False 属正常；
# 核心断言：点开始训练后轮次与进度真实推进
if act2[1] < act1[1] or act2[2] < act1[2]:
    ok = False
if act2[1] <= 0:
    ok = False
    print("⚠ 训练轮次未推进（可能客户端连接失败）")

# 清理
try:
    win.stop_training()
    win.stop_all_clients()
    win.stop_server()
except Exception as e:
    print("清理:", e)

print("== 结果:", "✅ 训练流程符合预期（只有点开始训练后监控/进度才动）"
      if ok else "❌ 流程异常 ==")
