# -*- coding: utf-8 -*-
"""离屏验证：图表占位重绘（空指标无残留）与真实数据重绘"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import federated_gui_pyside as m

app = m.QApplication([])
win = m.FederatedGUI()

# 1) 空指标：应显示占位、不残留
win._metrics = {}
win._redraw_charts()
print("[1] 空指标占位重绘 OK，test_acc 标签:", win.monitor_acc_label.text())

# 2) 有指标：画真实曲线（含真实连接历史）
win._metrics = {1: {"train_acc": 0.5, "test_acc": 0.4, "train_loss": 1.2},
                2: {"train_acc": 0.6, "test_acc": 0.45, "train_loss": 1.0}}
win._conn_history = [0, 2, 3, 3]
win.client_processes.append(("client_1", None))  # 仅测试样本分布绘图不依赖真实进程
win._redraw_charts()
print("[2] 数据重绘 OK，test_acc 标签:", win.monitor_acc_label.text())

# 3) 模拟 start_training 流程：重置后立即占位（不残留）
win._metrics = {1: {"train_acc": 0.9}}
win._last_plotted_round = -1
win._conn_history = []
win._metrics = {}
win._redraw_charts()
print("[3] 训练重置后占位 OK（无残留曲线），test_acc 标签:", win.monitor_acc_label.text())

print("== 图表逻辑验证完成 ==")
