# -*- coding: utf-8 -*-
"""离屏验证：PySide6 版页面映射（错位修复后，导航索引必须与页面一一对应）"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import federated_gui_pyside as m

app = m.QApplication([])
win = m.FederatedGUI()

print("页面总数(stack.count):", win.stack.count(), "(应为 10，含外观设置)")
print("初始页 currentIndex:", win.stack.currentIndex(), "(应为 0=控制面板)")
ok = True
for idx, (key, title, _emoji) in enumerate(win._nav_defs):
    win._switch_page(idx)
    actual = win.stack.currentIndex()
    match = "OK" if actual == idx else "❌ 错位"
    if actual != idx:
        ok = False
    print(f"导航[{idx}] {title}: 页面={actual} {match}")
    # 页头标题应与导航对应（每个页面构建时都更新 page_title）
    print(f"    页头: {win.page_title.text()}")
print("== 页面映射", "全部正确 ✓" if ok else "存在错位 ✗", "==")
