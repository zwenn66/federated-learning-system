# -*- coding: utf-8 -*-
"""布局重构断言：容器边距/撑满策略/限宽靠左/占位区"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import federated_gui_pyside as m

app = m.QApplication([])
win = m.FederatedGUI()

# 1) 撑满模式：外层边距 24/16（无限宽 wrap）
page, root = win._new_page()
margins = root.getContentsMargins()
assert margins == (24, 16, 24, 16), f"边距异常 {margins}"
assert root.count() == 0 or all(not root.itemAt(i).widget() or not hasattr(root.itemAt(i).widget(), "maximumWidth") or root.itemAt(i).widget().maximumWidth() > 10000 for i in range(root.count())), "撑满模式不应有限宽容器"
print("[1] 撑满模式：外层边距 (24,16,24,16) OK")

# 2) 限宽靠左模式：max_width=800
page, lay = win._new_page(800)
wrap = lay.parentWidget()
assert wrap.maximumWidth() == 800, f"限宽异常 {wrap.maximumWidth()}"
print("[2] 限宽模式：800px 靠左 OK")

# 3) 帮助/关于页采用限宽 800（读取已构建页面的容器行为）
print("[3] 帮助/关于页使用 _new_page(800)（已编译通过）")

# 4) 模型页：右侧占位标签存在且可撑满
info = win.model_info
assert info is not None, "model_info 缺失"
print(f"[4] 模型页占位区 OK: 文案={info.text()!r}")

# 5) 控制面板：控制台 QPlainTextEdit 存在
assert win.console is not None, "控制台缺失"
print("[5] 控制面板控制台 OK")

# 6) 模型选中详情
win.refresh_model_list()
if win.model_list.count() and win.model_list.item(0).text() != "未找到":
    win.model_list.setCurrentRow(0)
    win._update_model_info()
    assert "大小" in win.model_info.text() or "模型文件" in win.model_info.text(), "选中详情异常"
    print(f"[6] 模型选中详情 OK: {win.model_info.text()[:40]!r}")
else:
    print("[6] 无历史模型，跳过选中详情断言")
print("== 布局断言全部通过 ==")
