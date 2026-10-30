#!/usr/bin/env python3
"""联邦学习系统 v2.0 —— PySide6 专业版 GUI

技术要点（对应三条 UI 要求）：
1. 侧边栏：QScrollArea 纵向滚动 + 合理 minimumWidth，文字与图标完整包容
2. 折叠功能：导航栏顶部汉堡按钮 + QPropertyAnimation 平滑过渡，收起仅图标、展开恢复全宽
3. QSS 现代浅色科技配色：浅灰/淡蓝，Hover 悬停 + 选中 Active 左侧指示条，
   底部状态栏与导航栏视觉区分但风格一致

工程逻辑与 tkinter 版保持一致（真实数据，不造假）：
- 子进程输出全部 DEVNULL（避免管道阻塞导致训练卡死）
- 训练进度/指标来自 server.log；客户端样本数/准确率来自 client_logs/
- 单实例文件锁（.gui.lock），防重复启动
"""
import os
import sys
import json
import time
import re
import subprocess
import msvcrt
from datetime import datetime

# matplotlib 使用 Qt 后端（必须在 pyplot 导入前设置）
import matplotlib
matplotlib.use("QtAgg")
# matplotlib 中文字体（避免图表中文乱码）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtGui import QFont, QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFrame, QLabel, QPushButton, QLineEdit, QSpinBox,
    QComboBox, QHBoxLayout, QVBoxLayout, QGridLayout, QScrollArea, QStackedWidget,
    QProgressBar, QPlainTextEdit, QListWidget, QListWidgetItem, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QSizePolicy, QApplication,
    QButtonGroup, QSplitter, QStatusBar, QHeaderView,
)

APP_TITLE = "联邦学习系统 v2.0"
NAV_EXPANDED = 208
NAV_COLLAPSED = 62
ANIM_MS = 220

# ============ 统一配色常量 ============
C_PRIMARY = "#2b5fd9"       # 主蓝（科技淡蓝系）
C_PRIMARY_BG = "#e3edfd"    # 主蓝淡底（选中）
C_HOVER_BG = "#dde7f5"      # 悬停淡底
C_SIDEBAR_BG = "#eef2f7"    # 侧边栏浅灰（略带科技淡蓝）
C_CONTENT_BG = "#f5f7fa"    # 内容区浅灰
C_TEXT = "#2c3e50"          # 主文字深灰
C_MUTED = "#7f8c8d"         # 弱化文字
C_BORDER = "#dde3ec"        # 边框
C_SUCCESS = "#27ae60"
C_DANGER = "#c0392b"
C_CHART_BG = "#f5f7fa"

# ============ QSS：现代浅色科技风 ============
QSS_LIGHT = """
* { font-family: "Microsoft YaHei"; }
QMainWindow, #contentArea { background: %(contentBg)s; }
QStackedWidget { background: %(contentBg)s; }

/* ---------- 侧边导航 ---------- */
#sidebar { background: %(sidebarBg)s; border-right: 1px solid %(border)s; }
#appLogo { font-size: 16px; font-weight: 700; color: %(text)s; background: transparent; }
#appSub { font-size: 10px; color: %(muted)s; background: transparent; }
#collapseBtn { background: transparent; border: none; font-size: 18px; color: #4a6fa5; border-radius: 6px; padding: 4px 10px; }
#collapseBtn:hover { background: %(hoverBg)s; }
QScrollArea#navScroll { background: transparent; border: none; }
QWidget#navContainer { background: transparent; }

/* 导航项：文字左对齐、图标包容；Hover 淡底；选中左侧高亮指示条 */
QPushButton[nav="true"] {
    text-align: left;
    padding: 12px 14px;
    margin: 2px 10px;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    color: %(text)s;
    background: transparent;
}
QPushButton[nav="true"]:hover { background: %(hoverBg)s; color: #1f2d3d; }
QPushButton[nav="true"]:checked {
    background: %(primaryBg)s;
    color: %(primary)s;
    border-left: 3px solid %(primary)s;
    font-weight: 600;
}
QPushButton[nav="true"][collapsed="true"] {
    text-align: center;
    padding: 12px 0;
    margin: 2px 8px;
    font-size: 16px;
}
#sidebarFooter { background: transparent; color: %(muted)s; font-size: 10px; }

/* ---------- 内容区控件 ---------- */
QFrame#pageCard {
    background: #ffffff;
    border: 1px solid %(border)s;
    border-radius: 8px;
}
QLabel[pageTitle="true"] { font-size: 18px; font-weight: 700; color: %(text)s; }
QLabel[pageSub="true"] { font-size: 12px; color: %(muted)s; }
QLabel[kpi="true"] { font-size: 26px; font-weight: 700; color: %(text)s; }
QLabel[kpiLabel="true"] { font-size: 12px; color: %(muted)s; }

QPushButton {
    background: #ffffff;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    color: %(text)s;
}
QPushButton:hover { background: %(hoverBg)s; border-color: %(primary)s; }
QPushButton:disabled { color: #b5bfcc; background: #f0f3f8; border-color: %(border)s; }
QPushButton#btnPrimary { background: #ffffff; color: %(primary)s; border: 1px solid %(primary)s; font-weight: 600; }
QPushButton#btnPrimary:hover { background: %(primaryBg)s; color: %(primary)s; }
QPushButton#btnPrimary:disabled { background: %(primary)s; color: #ffffff; border: none; }
QPushButton#btnDanger { background: #fdeeee; color: %(danger)s; border: 1px solid #f5c6c6; }
QPushButton#btnDanger:hover { background: #fbdcdc; }

QLineEdit, QSpinBox, QComboBox {
    background: #ffffff;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 13px;
    color: %(text)s;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color: %(primary)s; }

QProgressBar {
    border: 1px solid %(border)s;
    border-radius: 6px;
    background: #eef1f6;
    text-align: center;
    font-size: 12px;
    color: %(text)s;
    height: 18px;
}
QProgressBar::chunk { background: %(primary)s; border-radius: 5px; }

QPlainTextEdit, QListWidget, QTableWidget {
    background: #ffffff;
    border: 1px solid %(border)s;
    border-radius: 6px;
    font-size: 12px;
    color: %(text)s;
}
QHeaderView::section {
    background: %(sidebarBg)s;
    border: none;
    border-bottom: 1px solid %(border)s;
    padding: 8px;
    font-size: 12px;
    font-weight: 600;
    color: %(text)s;
}
QTableWidget { gridline-color: %(border)s; }
QTableWidget::item { padding: 4px 6px; }
QTableWidget::item:selected { background: %(primaryBg)s; color: %(primary)s; }

/* ---------- 底部状态栏（与导航区分但风格一致） ---------- */
QStatusBar {
    background: #f1f4f9;
    border-top: 1px solid %(border)s;
    color: %(muted)s;
    font-size: 12px;
}
QStatusBar QLabel { color: %(muted)s; padding: 0 8px; }
QStatusBar QLabel[status="ok"] { color: %(success)s; }
QStatusBar QLabel[status="bad"] { color: %(danger)s; }

QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #c3cddb; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: %(primary)s; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
""" % {
    "contentBg": C_CONTENT_BG, "sidebarBg": C_SIDEBAR_BG, "border": C_BORDER,
    "text": C_TEXT, "muted": C_MUTED, "hoverBg": C_HOVER_BG,
    "primary": C_PRIMARY, "primaryBg": C_PRIMARY_BG,
    "success": C_SUCCESS, "danger": C_DANGER,
}

# 深色 QSS（可选主题）
QSS_DARK = """
* { font-family: "Microsoft YaHei"; }
QMainWindow, #contentArea, QStackedWidget { background: #16181d; }
#sidebar { background: #1c2130; border-right: 1px solid #262d3d; }
#appLogo { font-size: 16px; font-weight: 700; color: #e8eef5; }
#appSub { font-size: 10px; color: #9db1c4; }
#collapseBtn { background: transparent; border: none; font-size: 18px; color: #8fb3e8; }
#collapseBtn:hover { background: #262d3d; }
QScrollArea#navScroll { background: transparent; border: none; }
QWidget#navContainer { background: transparent; }
QPushButton[nav="true"] {
    text-align: left; padding: 12px 14px; margin: 2px 10px; border: none; border-radius: 6px;
    font-size: 13px; color: #c6d2e0; background: transparent;
}
QPushButton[nav="true"]:hover { background: #262d3d; color: #ffffff; }
QPushButton[nav="true"]:checked { background: #243249; color: #7aa5f0; border-left: 3px solid #7aa5f0; font-weight: 600; }
QPushButton[nav="true"][collapsed="true"] { text-align: center; padding: 12px 0; margin: 2px 8px; font-size: 16px; }
#sidebarFooter { color: #5f7188; font-size: 10px; }
QFrame#pageCard { background: #1e222b; border: 1px solid #2a3140; border-radius: 8px; }
QLabel[pageTitle="true"] { font-size: 18px; font-weight: 700; color: #e8eef5; }
QLabel[pageSub="true"] { font-size: 12px; color: #9db1c4; }
QLabel[kpi="true"] { font-size: 26px; font-weight: 700; color: #e8eef5; }
QLabel[kpiLabel="true"] { font-size: 12px; color: #9db1c4; }
QPushButton { background: #262d3d; border: 1px solid #2a3140; border-radius: 6px; padding: 8px 16px; font-size: 13px; color: #c6d2e0; }
QPushButton:hover { background: #303a4e; border-color: #7aa5f0; }
QPushButton:disabled { color: #5a6a80; background: #20242e; }
QPushButton#btnPrimary { background: #2a3b55; color: #6ea8fe; border: 1px solid #3b6fd4; font-weight: 600; }
QPushButton#btnPrimary:hover { background: #33476a; }
QPushButton#btnPrimary:disabled { background: #3b6fd4; color: #ffffff; border: none; }
QPushButton#btnDanger { background: #3a2024; color: #e88; border: 1px solid #5c2a2a; }
QLineEdit, QSpinBox, QComboBox { background: #20242e; border: 1px solid #2a3140; border-radius: 6px; padding: 7px 10px; font-size: 13px; color: #c6d2e0; }
QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color: #7aa5f0; }
QProgressBar { border: 1px solid #2a3140; border-radius: 6px; background: #20242e; text-align: center; font-size: 12px; color: #c6d2e0; height: 18px; }
QProgressBar::chunk { background: #3b6fd4; border-radius: 5px; }
QPlainTextEdit, QListWidget, QTableWidget { background: #1e222b; border: 1px solid #2a3140; border-radius: 6px; font-size: 12px; color: #c6d2e0; }
QHeaderView::section { background: #20242e; border: none; border-bottom: 1px solid #2a3140; padding: 8px; font-size: 12px; font-weight: 600; color: #c6d2e0; }
QTableWidget { gridline-color: #2a3140; }
QTableWidget::item:selected { background: #243249; color: #7aa5f0; }
QStatusBar { background: #1c2130; border-top: 1px solid #262d3d; color: #9db1c4; font-size: 12px; }
QStatusBar QLabel { color: #9db1c4; padding: 0 8px; }
QStatusBar QLabel[status="ok"] { color: #4cc38a; }
QStatusBar QLabel[status="bad"] { color: #e88; }
QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar::handle:vertical { background: #3a4356; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #7aa5f0; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def _acquire_single_instance_lock():
    """Windows 文件锁单实例（进程退出自动释放，无 TIME_WAIT 误报）"""
    lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gui.lock")
    try:
        f = open(lock_path, "a+")
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return f
        except OSError:
            f.close()
            return None
    except Exception:
        return None


def _load_config():
    default = {"server_port": 8080, "num_rounds": 20, "min_clients": 2,
               "local_epochs": 1, "model_save_path": "./saved_models"}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_config.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in default.items():
                cfg.setdefault(k, v)
            return cfg
    except Exception:
        pass
    return default


class NavButton(QPushButton):
    """侧边栏导航项：emoji + 文字，折叠时仅 emoji"""

    def __init__(self, emoji: str, text: str, parent=None):
        super().__init__(parent)
        self._emoji = emoji
        self._full_text = text
        self.setProperty("nav", True)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setText(f"{emoji}  {text}")

    def set_collapsed(self, collapsed: bool):
        self.setProperty("collapsed", collapsed)
        self.setText(self._emoji if collapsed else f"{self._emoji}  {self._full_text}")
        self.style().unpolish(self)
        self.style().polish(self)


class FederatedGUI(QMainWindow):
    """联邦学习系统 PySide6 专业版主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 760)
        self.setMinimumSize(1120, 700)

        self.config = _load_config()
        self.server_process = None
        self.client_processes = []  # [(client_id, Popen)]
        self.is_training = False
        self.current_round = 0
        self.total_rounds = int(self.config["num_rounds"])
        self._collapsed = False
        self._metrics = {}  # round -> {train_acc, train_loss, test_acc}
        # 会话边界：以 GUI 打开时刻的 server.log 为准判定"旧日志残留"，
        # 而非点击开始训练时（此时训练可能已自动开始写入新轮次，避免误判）
        self._session_had_history = self._peek_round() >= 1
        self._session_click_round = -1
        self._session_new_round_seen = False

        self._build_ui()
        self._build_nav()
        self._build_pages()
        self._build_statusbar()
        self.apply_theme("light")

        # 定时刷新
        self._monitor_timer = QTimer(self)
        self._monitor_timer.timeout.connect(self._refresh_status)
        self._monitor_timer.start(3000)

        self._train_timer = QTimer(self)
        self._train_timer.timeout.connect(self._refresh_training)
        self._train_timer.start(2000)

        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._refresh_log)
        self._log_timer.start(3000)

        self._refresh_log()
        self._refresh_status()

    # ================= 界面骨架 =================
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("contentArea")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 侧边栏
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(NAV_EXPANDED)
        root.addWidget(self.sidebar)

        # 内容区（统一页头 + 页面栈）
        content = QWidget()
        content.setObjectName("contentArea")
        cv = QVBoxLayout(content)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        header = QWidget()
        hl = QVBoxLayout(header)
        hl.setContentsMargins(20, 14, 20, 4)
        self.page_title = QLabel("联邦学习系统 - 控制面板")
        self.page_title.setProperty("pageTitle", True)
        self.page_sub = QLabel("服务器 · 客户端 · 训练 统一控制")
        self.page_sub.setProperty("pageSub", True)
        hl.addWidget(self.page_title)
        hl.addWidget(self.page_sub)
        cv.addWidget(header, 0)

        self.stack = QStackedWidget()
        self.stack.setObjectName("contentArea")
        cv.addWidget(self.stack, 1)
        root.addWidget(content, 1)

    def _build_nav(self):
        sb = QVBoxLayout(self.sidebar)
        sb.setContentsMargins(8, 12, 8, 8)
        sb.setSpacing(4)

        # 顶部：折叠按钮 + 应用名
        head = QHBoxLayout()
        head.setSpacing(6)
        self.collapse_btn = QPushButton("☰")
        self.collapse_btn.setObjectName("collapseBtn")
        self.collapse_btn.setToolTip("折叠 / 展开导航")
        self.collapse_btn.setFixedWidth(38)
        self.collapse_btn.clicked.connect(self._toggle_sidebar)
        head.addWidget(self.collapse_btn)
        self.app_logo = QLabel("联邦学习系统")
        self.app_logo.setObjectName("appLogo")
        head.addWidget(self.app_logo, 1)
        sb.addLayout(head)

        sub = QLabel("v2.0 · PySide6 专业版")
        sub.setObjectName("appSub")
        sub.setContentsMargins(6, 0, 0, 6)
        sb.addWidget(sub)

        # 滚动导航区（QScrollArea，内容过多时纵向滚动）
        self.nav_scroll = QScrollArea()
        self.nav_scroll.setObjectName("navScroll")
        self.nav_scroll.setWidgetResizable(True)
        self.nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        nav_container = QWidget()
        nav_container.setObjectName("navContainer")
        self._nav_layout = QVBoxLayout(nav_container)
        self._nav_layout.setContentsMargins(0, 4, 0, 4)
        self._nav_layout.setSpacing(0)
        self._nav_layout.addStretch(1)
        self.nav_scroll.setWidget(nav_container)
        sb.addWidget(self.nav_scroll, 1)

        # 侧边栏底部版本
        self.sidebar_footer = QLabel("联邦学习\n安全加固版")
        self.sidebar_footer.setObjectName("sidebarFooter")
        self.sidebar_footer.setAlignment(Qt.AlignCenter)
        sb.addWidget(self.sidebar_footer)

        # 折叠动画
        self._anim = QPropertyAnimation(self.sidebar, b"maximumWidth", self)
        self._anim.setDuration(ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._anim.finished.connect(self._after_collapse)

        # 导航项定义：emoji, 文字, 页面索引
        self._nav_defs = [
            ("🏠", "控制面板", 0),
            ("🎨", "外观设置", 1),
            ("📊", "训练监控", 2),
            ("🧠", "模型管理", 3),
            ("📋", "系统日志", 4),
            ("⚙️", "服务器状态", 5),
            ("👥", "客户端状态", 6),
            ("📈", "性能分析", 7),
            ("❓", "帮助文档", 8),
            ("ℹ️", "关于系统", 9),
        ]
        self._nav_buttons = []
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for emoji, text, idx in self._nav_defs:
            btn = NavButton(emoji, text)
            btn.clicked.connect(lambda checked, i=idx: self._switch_page(i))
            self._nav_layout.insertWidget(self._nav_layout.count() - 1, btn)
            self._nav_group.addButton(btn)
            self._nav_buttons.append(btn)
        if self._nav_buttons:
            self._nav_buttons[0].setChecked(True)

    def _toggle_sidebar(self):
        if self._collapsed:
            self._anim.setStartValue(NAV_COLLAPSED)
            self._anim.setEndValue(NAV_EXPANDED)
            self.collapse_btn.setText("☰")
        else:
            self._anim.setStartValue(NAV_EXPANDED)
            self._anim.setEndValue(NAV_COLLAPSED)
            self.collapse_btn.setText("☰")
        self._anim.start()

    def _after_collapse(self):
        self._collapsed = not self._collapsed
        for btn in self._nav_buttons:
            btn.set_collapsed(self._collapsed)
        if self._collapsed:
            self.app_logo.setText("")
            self.sidebar_footer.setText("FL")
        else:
            self.app_logo.setText("联邦学习系统")
            self.sidebar_footer.setText("联邦学习\n安全加固版")
        self.collapse_btn.setToolTip("展开导航" if self._collapsed else "折叠导航")

    def _switch_page(self, idx: int):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == idx)
        # 页面标题联动
        titles = ["控制面板", "外观设置", "训练监控", "模型管理", "系统日志",
                  "服务器状态", "客户端状态", "性能分析", "帮助文档", "关于系统"]
        subs = ["服务器 · 客户端 · 训练 统一控制",
                "界面主题与视觉风格",
                "实时训练曲线与收敛进度",
                "全局模型文件管理",
                "服务器运行日志（server.log）",
                "服务器运行状态与实时指标",
                "各客户端连接状态与本地指标",
                "系统运行性能分析",
                "使用说明",
                "版本与致谢"]
        self.page_title.setText(f"联邦学习系统 - {titles[idx]}")
        self.page_sub.setText(subs[idx])

    # ================= 页面构建 =================
    def _build_pages(self):
        self._build_control_page()
        self._build_appearance_page()
        self._build_monitor_page()
        self._build_model_page()
        self._build_log_page()
        self._build_server_status_page()
        self._build_client_status_page()
        self._build_perf_page()
        self._build_help_page()
        self._build_about_page()
        self.stack.setCurrentIndex(0)

    def _new_page(self, max_width: int = 0):
        """创建页面容器：
        - 默认(max_width=0)：内容撑满右侧可用区域（卡片 QSizePolicy.Expanding），
          外层统一保留 24px 边距、行距 12px，充分利用宽屏空间
        - max_width>0：内容限宽并靠左对齐（适合帮助/关于等长文本页面，避免超长行）"""
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)
        if max_width and max_width < 100000:
            wrap = QWidget()
            wrap.setMaximumWidth(max_width)
            wrap.setMinimumWidth(320)
            lay = QVBoxLayout(wrap)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(12)
            root.addWidget(wrap)
            root.addStretch(1)
            return page, lay
        return page, root

    def _card(self, title: str):
        frame = QFrame()
        frame.setObjectName("pageCard")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(12)
        if title:
            t = QLabel(title)
            t.setStyleSheet("font-size:13px;font-weight:600;color:#5a6b7d;")
            lay.addWidget(t)
        return frame, lay

    # ----- 控制面板 -----
    def _build_control_page(self):
        page, root = self._new_page()

        # 服务器配置
        card, lay = self._card("服务器配置")
        lay.setContentsMargins(16, 20, 16, 16)  # 内部四周留白，顶部加大呼吸感
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)  # 行距 12px
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 2)

        def _cfg_label(text: str):
            lab = QLabel(text)
            lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return lab

        grid.addWidget(_cfg_label("服务器端口:"), 0, 0)
        self.port_edit = QLineEdit(str(self.config["server_port"]))
        self.port_edit.setFixedHeight(34)
        grid.addWidget(self.port_edit, 0, 1)
        grid.addWidget(_cfg_label("训练轮数:"), 0, 2)
        self.rounds_edit = QLineEdit(str(self.config["num_rounds"]))
        self.rounds_edit.setFixedHeight(34)
        grid.addWidget(self.rounds_edit, 0, 3)
        grid.addWidget(_cfg_label("最小客户端数:"), 1, 0)
        self.min_clients_edit = QLineEdit(str(self.config["min_clients"]))
        self.min_clients_edit.setFixedHeight(34)
        grid.addWidget(self.min_clients_edit, 1, 1)
        grid.addWidget(_cfg_label("本地训练轮数:"), 1, 2)
        self.local_epochs_edit = QLineEdit(str(self.config["local_epochs"]))
        self.local_epochs_edit.setFixedHeight(34)
        grid.addWidget(self.local_epochs_edit, 1, 3)
        grid.addWidget(_cfg_label("模型路径:"), 2, 0)
        self.model_path_edit = QLineEdit(self.config["model_save_path"])
        self.model_path_edit.setFixedHeight(34)
        # 只占列 1：右端与上方「最小客户端数」输入框右端垂直对齐
        grid.addWidget(self.model_path_edit, 2, 1)
        browse_btn = QPushButton("浏览...")
        browse_btn.setFixedHeight(34)
        browse_btn.clicked.connect(self._browse_model_path)
        # 单独占列 3：右端与上方「本地训练轮数」输入框右端对齐
        grid.addWidget(browse_btn, 2, 3)
        lay.addLayout(grid)
        lay.addSpacing(15)  # 按钮行与输入框之间的明确空隙

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_start_server = QPushButton("🚀 启动服务器")
        self.btn_start_server.setObjectName("btnPrimary")
        self.btn_start_server.clicked.connect(self.start_server)
        self.btn_stop_server = QPushButton("🛑 停止服务器")
        self.btn_stop_server.clicked.connect(self.stop_server)
        self.btn_stop_server.setEnabled(False)
        save_btn = QPushButton("💾 保存配置")
        save_btn.clicked.connect(self._save_config)
        test_btn = QPushButton("🔍 测试端口")
        test_btn.clicked.connect(self._test_port)
        for b in (self.btn_start_server, self.btn_stop_server, save_btn, test_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        root.addWidget(card)

        # 客户端管理
        card, lay = self._card("客户端管理")
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("客户端数量:"))
        self.client_count_spin = QSpinBox()
        self.client_count_spin.setRange(1, 10)
        self.client_count_spin.setValue(3)
        row.addWidget(self.client_count_spin)
        self.btn_start_clients = QPushButton("▶️ 启动客户端")
        self.btn_start_clients.clicked.connect(self.start_clients)
        self.btn_stop_clients = QPushButton("⏹️ 停止所有客户端")
        self.btn_stop_clients.clicked.connect(self.stop_all_clients)
        self.btn_stop_clients.setEnabled(False)
        test_clients_btn = QPushButton("🧪 测试客户端")
        test_clients_btn.clicked.connect(self._test_clients)
        for b in (self.btn_start_clients, self.btn_stop_clients, test_clients_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.client_summary = QLabel("客户端: 0 个未启动")
        self.client_summary.setProperty("pageSub", True)
        lay.addWidget(self.client_summary)
        root.addWidget(card)

        # 训练控制
        card, lay = self._card("训练控制")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_start_train = QPushButton("🚀 开始训练")
        self.btn_start_train.setObjectName("btnPrimary")
        self.btn_start_train.clicked.connect(self.start_training)
        self.btn_start_train.setEnabled(False)
        self.btn_pause_train = QPushButton("⏸️ 暂停训练")
        self.btn_pause_train.clicked.connect(self.pause_training)
        self.btn_pause_train.setEnabled(False)
        self.btn_stop_train = QPushButton("⏹️ 停止训练")
        self.btn_stop_train.clicked.connect(self.stop_training)
        self.btn_stop_train.setEnabled(False)
        for b in (self.btn_start_train, self.btn_pause_train, self.btn_stop_train):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.train_status = QLabel("等待开始训练...")
        self.train_status.setProperty("pageSub", True)
        lay.addWidget(self.train_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        lay.addWidget(self.progress)
        root.addWidget(card)

        # 控制台日志：垂直拉伸填满窗口底部剩余空间
        card, lay = self._card("控制台输出（server.log）")
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(2000)
        lay.addWidget(self.console, 1)
        root.addWidget(card, 1)

        self.stack.addWidget(page)

    # ----- 外观设置 -----
    def _build_appearance_page(self):
        page, root = self._new_page()

        # 主题风格
        card, lay = self._card("主题风格")
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(QLabel("界面主题:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["浅色科技（默认）", "深色专业"])
        self.theme_combo.currentIndexChanged.connect(
            lambda i: self.apply_theme("dark" if i == 1 else "light"))
        row.addWidget(self.theme_combo)
        row.addStretch(1)
        lay.addLayout(row)
        desc = QLabel(
            "浅色科技：浅灰带淡蓝的侧边栏 + 白色内容区，Hover 悬停 + Active 左侧指示条"
            "（参考 Ant Design / Element UI 极简风格）\n"
            "深色专业：深蓝灰侧边栏 + 深色内容区，低饱和配色，适合长时间盯屏")
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size:13px;color:#5a6b7d;")
        lay.addWidget(desc)
        root.addWidget(card)

        # 侧边栏
        card, lay = self._card("侧边栏")
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(QLabel("当前模式:"))
        self.sidebar_mode_lbl = QLabel("完整宽度（208px）")
        self.sidebar_mode_lbl.setStyleSheet("font-size:13px;color:#2b5fd9;font-weight:600;")
        row.addWidget(self.sidebar_mode_lbl)
        row.addStretch(1)
        lay.addLayout(row)
        tip = QLabel("点击导航栏顶部 ☰ 按钮，可平滑折叠为仅图标模式（62px），再次点击展开")
        tip.setWordWrap(True)
        tip.setStyleSheet("font-size:13px;color:#5a6b7d;")
        lay.addWidget(tip)
        root.addWidget(card)
        root.addStretch(1)

        self.stack.addWidget(page)

    # ----- 训练监控 -----
    def _build_monitor_page(self):
        page, root = self._new_page()

        top = QHBoxLayout()
        top.addWidget(QLabel("训练进度:"))
        self.monitor_progress = QProgressBar()
        self.monitor_progress.setRange(0, 100)
        top.addWidget(self.monitor_progress, 1)
        self.monitor_round_label = QLabel("轮次: 0/0")
        top.addWidget(self.monitor_round_label)
        self.monitor_acc_label = QLabel("test_acc: —")
        top.addWidget(self.monitor_acc_label)
        root.addLayout(top)

        # matplotlib 2x2
        self.fig = plt.Figure(figsize=(9.5, 6.2), dpi=100)
        self.fig.patch.set_facecolor(C_CHART_BG)
        gs = self.fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.grid(True, alpha=0.3)
        self.ax1.set_title("Accuracy Trend")
        self.ax1.set_xlabel("训练轮数")
        self.ax1.set_ylabel("准确率")
        self.ax2.set_title("Loss Trend")
        self.ax2.set_xlabel("训练轮数")
        self.ax2.set_ylabel("损失值")
        self.ax3.set_title("Connected Clients")
        self.ax3.set_xlabel("时间")
        self.ax3.set_ylabel("客户端数量")
        self.ax4.set_title("Sample Distribution (Non-IID)")
        self.ax4.set_xlabel("客户端")
        self.ax4.set_ylabel("样本数量")
        self.ax1.text(0.5, 0.5, "等待训练数据…\n启动服务器并开始训练后自动更新",
                      ha="center", va="center", transform=self.ax1.transAxes,
                      fontsize=10, color=C_MUTED)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.canvas, 1)

        self.stack.addWidget(page)

    # ----- 模型管理 -----
    def _build_model_page(self):
        page, root = self._new_page()
        hbox = QHBoxLayout()
        hbox.setSpacing(10)

        card, lay = self._card("已保存模型")
        self.model_list = QListWidget()
        lay.addWidget(self.model_list, 1)
        hbox.addWidget(card, 3)

        card, lay = self._card("操作")
        row = QHBoxLayout()
        refresh_btn = QPushButton("🔄 刷新列表")
        refresh_btn.clicked.connect(self.refresh_model_list)
        load_btn = QPushButton("📥 加载模型")
        load_btn.clicked.connect(self.load_selected_model)
        delete_btn = QPushButton("🗑️ 删除模型")
        delete_btn.setObjectName("btnDanger")
        delete_btn.clicked.connect(self.delete_selected_model)
        export_btn = QPushButton("📤 导出模型")
        export_btn.clicked.connect(self.export_selected_model)
        for b in (refresh_btn, load_btn, delete_btn, export_btn):
            row.addWidget(b)
        lay.addLayout(row)
        self.model_info = QLabel("请从左侧选择模型")
        self.model_info.setWordWrap(True)
        self.model_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_info.setStyleSheet(
            "font-size:15px;color:#a6b0bc;border:1px dashed #d3dae3;border-radius:6px;"
            "padding:24px;background:#f7f9fb;")
        lay.addWidget(self.model_info, 1)
        lay.addStretch(1)
        hbox.addWidget(card, 1)
        root.addLayout(hbox)

        self.refresh_model_list()
        self.stack.addWidget(page)

    # ----- 系统日志 -----
    def _build_log_page(self):
        page, root = self._new_page()
        card, lay = self._card("服务器运行日志（server.log，实时）")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        lay.addWidget(self.log_view, 1)
        root.addWidget(card)
        self.stack.addWidget(page)

    # ----- 服务器状态 -----
    def _build_server_status_page(self):
        page, root = self._new_page()
        card, lay = self._card("实时状态")
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(14)
        self._sv_labels = {}
        for i, (k, key) in enumerate([
            ("服务器状态", "server"), ("运行端口", "port"),
            ("训练状态", "train"), ("客户端数量", "clients"),
            ("当前轮数", "round"), ("训练进度", "progress"),
        ]):
            lab = QLabel(k)
            lab.setStyleSheet("font-weight:600;color:#5a6b7d;")
            val = QLabel("—")
            val.setProperty("kpi", True)
            val.setStyleSheet("font-size:20px;font-weight:700;color:#2c3e50;")
            self._sv_labels[key] = val
            grid.addWidget(lab, i // 2 * 2, i % 2 * 2)
            grid.addWidget(val, i // 2 * 2, i % 2 * 2 + 1)
            grid.setColumnStretch(i % 2 * 2 + 1, 1)
        lay.addLayout(grid)
        lay.addStretch(1)
        root.addWidget(card)
        self.stack.addWidget(page)

    # ----- 客户端状态 -----
    def _build_client_status_page(self):
        page, root = self._new_page()
        card, lay = self._card("已连接的客户端（真实进程与本地指标）")
        self.client_table = QTableWidget(0, 5)
        self.client_table.setHorizontalHeaderLabels(["ID", "状态", "PID", "样本数", "准确率"])
        # 全列 Stretch：列宽随窗口大小自适应，表头与数据列严格对齐
        self.client_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.client_table.verticalHeader().setVisible(False)
        self.client_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.client_table.setSelectionBehavior(QTableWidget.SelectRows)
        lay.addWidget(self.client_table, 1)
        row = QHBoxLayout()
        refresh_btn = QPushButton("🔄 刷新列表")
        refresh_btn.clicked.connect(self._refresh_client_table)
        row.addWidget(refresh_btn)
        row.addStretch(1)
        lay.addLayout(row)
        root.addWidget(card)
        self.stack.addWidget(page)

    # ----- 性能分析 -----
    def _info_group(self, lay, title: str, items, tip: str = ""):
        """分组说明：小标题 + QGridLayout 对齐条目（标签固定宽、内容自动换行）"""
        t = QLabel(title)
        t.setStyleSheet("font-size:14px;font-weight:700;color:#2b5fd9;padding-top:6px;")
        lay.addWidget(t)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        for i, (head, body) in enumerate(items):
            h = QLabel(head)
            h.setStyleSheet("font-size:14px;font-weight:600;color:#3d5167;")
            h.setFixedWidth(150)
            h.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            b = QLabel(body)
            b.setStyleSheet("font-size:14px;color:#5a6b7d;line-height:22px;")
            b.setWordWrap(True)
            b.setMaximumWidth(880)  # 撑满窗口时保持合理阅读宽度，页面协调
            grid.addWidget(h, i, 0)
            grid.addWidget(b, i, 1)
        lay.addLayout(grid)
        if tip:
            tip_lbl = QLabel(tip)
            tip_lbl.setStyleSheet("font-size:13px;color:#8a97a8;padding-top:4px;")
            tip_lbl.setWordWrap(True)
            lay.addWidget(tip_lbl)

    # ----- 性能 / 帮助 / 关于 -----
    def _build_perf_page(self):
        page, root = self._new_page()
        # 真实训练指标（每 3 秒由 _refresh_perf 刷新）
        card, lay = self._card("训练指标（实时真实数据）")
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(14)
        self._perf_labels = {}
        for i, (k, key) in enumerate([
            ("训练轮次", "round"), ("训练准确率", "train_acc"),
            ("测试准确率", "test_acc"), ("训练损失", "train_loss"),
            ("客户端数量", "clients"), ("客户端样本总数", "samples"),
        ]):
            lab = QLabel(k)
            lab.setStyleSheet("font-weight:600;color:#5a6b7d;")
            val = QLabel("—")
            val.setStyleSheet("font-size:20px;font-weight:700;color:#2c3e50;")
            self._perf_labels[key] = val
            grid.addWidget(lab, i // 2 * 2, i % 2 * 2)
            grid.addWidget(val, i // 2 * 2, i % 2 * 2 + 1)
            grid.setColumnStretch(i % 2 * 2 + 1, 1)
        lay.addLayout(grid)
        root.addWidget(card)

        # 指标来源（补充说明）
        card, lay = self._card("指标说明")
        self._info_group(lay, "指标来源（真实训练数据）", [
            ("训练准确率", "由 server.log 每轮 FitMetrics 聚合指标解析"),
            ("测试准确率", "由 server.log 每轮 EvaluateMetrics 聚合指标解析"),
            ("客户端样本数", "Non-IID 划分后每客户端的本地训练样本数（最多 1000）"),
            ("客户端准确率", "客户端本地 fit / evaluate 写入 client_logs 的真实数值"),
        ], tip="提示：训练完成后，在「客户端状态」页可见每个客户端的样本数与准确率。")
        lay.addStretch(1)
        root.addWidget(card, 1)
        self.stack.addWidget(page)

    def _refresh_perf(self):
        """3 秒：性能分析页真实训练指标刷新"""
        if not hasattr(self, "_perf_labels"):
            return
        try:
            cur = self.current_round
            m = self._metrics.get(cur, {}) if self._metrics else {}
            self._perf_labels["round"].setText(f"{cur}/{self.total_rounds}")
            ta = m.get("train_acc")
            te = m.get("test_acc")
            tl = m.get("train_loss")
            if te is None and self._metrics:
                # 当前轮评估尚未完成时，回退显示最近已评估轮的 test_acc
                for r in range(cur, 0, -1):
                    v = self._metrics.get(r, {}).get("test_acc")
                    if v is not None:
                        te = v
                        break
            self._perf_labels["train_acc"].setText(
                f"{ta:.4f}" if ta is not None else "—")
            self._perf_labels["test_acc"].setText(
                f"{te:.4f}" if te is not None else "—")
            self._perf_labels["train_loss"].setText(
                f"{tl:.4f}" if tl is not None else "—")
            self._perf_labels["clients"].setText(str(len(self.client_processes)))
            total_samples = 0
            for cid, _ in self.client_processes:
                s, _a = self._read_client_metric(cid)
                if s:
                    total_samples += s
            self._perf_labels["samples"].setText(
                str(total_samples) if total_samples else "—")
        except Exception:
            pass

    def _build_help_page(self):
        page, root = self._new_page()
        card, lay = self._card("使用说明")
        self._info_group(lay, "操作步骤", [
            ("第 1 步", "启动服务器（请先确认端口未被占用）"),
            ("第 2 步", "启动客户端（建议数量 ≥ 2）"),
            ("第 3 步", "开始训练（客户端连接后也可能自动开始）"),
        ])
        self._info_group(lay, "常见问题", [
            ("端口冲突", "GUI 会弹窗提示，请更换端口或停掉占用进程"),
            ("日志解析", "训练进度 / 准确率实时解析自 server.log，无需手工刷新"),
            ("模型保存", "saved_models/ 目录下 global_model_round{N}_{ts}.h5"),
            ("客户端指标", "client_logs/ 目录按客户端 id 独立存储"),
            ("单实例限制", "重复启动会被文件锁拦截，请使用已打开的窗口"),
        ], tip="提示：模型管理页可加载 / 删除 / 导出历史模型。")
        lay.addStretch(1)
        root.addWidget(card)
        self.stack.addWidget(page)

    def _build_about_page(self):
        page, root = self._new_page()
        card, lay = self._card("关于")
        self._info_group(lay, "系统信息", [
            ("版本", "v2.0（PySide6 专业版）"),
            ("项目", "基于 Flower + TensorFlow 的 Fashion-MNIST Non-IID 联邦学习演示"),
            ("界面", "PySide6 + QSS 现代浅色科技风，QScrollArea 侧边栏 + 折叠动画"),
            ("图表", "Matplotlib（QtAgg 后端），中文字体已适配"),
            ("工程化", "真实指标上报、单实例文件锁、子进程管道防阻塞"),
        ])
        lay.addStretch(1)
        root.addWidget(card)
        self.stack.addWidget(page)

    # ----- 底部状态栏 -----
    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.sb_server = QLabel("服务器未运行")
        self.sb_server.setProperty("status", "bad")
        self.sb_clients = QLabel("客户端: 0")
        self.sb_train = QLabel("训练: 未开始")
        self.sb_round = QLabel("轮次: 0")
        sb.addWidget(QLabel("●"))
        sb.addWidget(self.sb_server)
        sb.addWidget(QLabel("|"))
        sb.addWidget(self.sb_clients)
        sb.addWidget(QLabel("|"))
        sb.addWidget(self.sb_train)
        sb.addWidget(QLabel("|"))
        sb.addWidget(self.sb_round)
        sb.addPermanentWidget(QLabel(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # ================= 主题 =================
    def apply_theme(self, name: str):
        self.setStyleSheet(QSS_DARK if name == "dark" else QSS_LIGHT)
        for btn in self._nav_buttons:
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ================= 工具函数 =================
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.console.appendPlainText(f"[{ts}] {msg}")

    def _browse_model_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型保存目录",
                                                self.model_path_edit.text())
        if path:
            self.model_path_edit.setText(path)

    def _save_config(self):
        try:
            self.config.update({
                "server_port": int(self.port_edit.text()),
                "num_rounds": int(self.rounds_edit.text()),
                "min_clients": int(self.min_clients_edit.text()),
                "local_epochs": int(self.local_epochs_edit.text()),
                "model_save_path": self.model_path_edit.text(),
            })
            with open("gui_config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.total_rounds = self.config["num_rounds"]
            self._log(f"配置已保存: 端口{self.config['server_port']} 轮数{self.total_rounds}")
            QMessageBox.information(self, "配置保存", "配置已保存，下次启动自动生效。")
        except ValueError as e:
            QMessageBox.warning(self, "配置保存", f"配置格式有误：{e}")

    def _test_port(self):
        port = int(self.port_edit.text())
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.5)
            ok = s.connect_ex(("127.0.0.1", port)) != 0
        if ok:
            QMessageBox.information(self, "端口检查", f"端口 {port} 可用")
            self._log(f"端口 {port} 可用")
        else:
            QMessageBox.warning(self, "端口检查", f"端口 {port} 已被占用，请使用其他端口")
            self._log(f"端口 {port} 已被占用")

    def _test_clients(self):
        if self.server_process is None or self.server_process.poll() is not None:
            QMessageBox.warning(self, "客户端测试", "服务器未运行，请先启动服务器")
            return
        QMessageBox.information(self, "客户端测试", "服务器正在运行，准备好接收客户端连接")

    def _venv_python(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(here, ".venv", "Scripts", "python.exe")
        return p if os.path.exists(p) else sys.executable

    # ================= 服务器 =================
    def start_server(self):
        try:
            port = int(self.port_edit.text())
            rounds = int(self.rounds_edit.text())
            min_clients = int(self.min_clients_edit.text())
            local_epochs = int(self.local_epochs_edit.text())
            model_path = self.model_path_edit.text() or "./saved_models"
        except ValueError as e:
            QMessageBox.warning(self, "启动服务器", f"配置格式有误：{e}")
            return

        # 端口占用检查
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                QMessageBox.warning(self, "启动服务器",
                                    f"端口 {port} 已被占用，请使用其他端口或先停止占用程序")
                return

        if self.server_process is not None and self.server_process.poll() is None:
            QMessageBox.warning(self, "启动服务器", "服务器已在运行，请先停止")
            return

        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "server_config_gui.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"port": port, "num_rounds": rounds, "min_fit_clients": min_clients,
                       "local_epochs": local_epochs, "model_save_path": model_path}, f,
                      indent=2, ensure_ascii=False)

        self.total_rounds = rounds
        self.is_training = False
        self.current_round = 0
        self._metrics = {}
        self._last_plotted_round = -1

        self.server_process = subprocess.Popen(
            [self._venv_python(), "federated_server.py", "--config", config_file],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.btn_start_server.setEnabled(False)
        self.btn_start_server.setText("🚀 服务器运行中")
        self.btn_stop_server.setEnabled(True)
        self.btn_start_clients.setEnabled(True)
        if hasattr(self, "progress"):
            self.progress.setValue(0)
            self.monitor_progress.setValue(0)
        self._conn_history = []
        self._redraw_charts()
        self._log(f"服务器已启动：端口 {port}，{rounds} 轮")
        self._refresh_status()

    def stop_server(self):
        if self.server_process is not None:
            try:
                self.server_process.kill()
            except Exception:
                pass
            self.server_process = None
        self.is_training = False
        self.btn_start_server.setEnabled(True)
        self.btn_start_server.setText("🚀 启动服务器")
        self.btn_stop_server.setEnabled(False)
        self.btn_start_clients.setEnabled(False)
        self.btn_start_train.setEnabled(False)
        self._log("服务器已停止")
        self._refresh_status()

    # ================= 客户端 =================
    def start_clients(self):
        n = self.client_count_spin.value()
        port = int(self.port_edit.text())
        if self.server_process is None or self.server_process.poll() is not None:
            QMessageBox.warning(self, "启动客户端", "请先启动服务器")
            return
        # 先停旧客户端，再按新数量启动（避免新旧混跑）
        if self.client_processes:
            self._log(f"检测到 {len(self.client_processes)} 个旧客户端，先停止再按 {n} 个启动")
            self.stop_all_clients()
        cwd = os.path.dirname(os.path.abspath(__file__))
        for i in range(n):
            cid = f"client_{i + 1}"
            proc = subprocess.Popen(
                [self._venv_python(), "federated_client.py",
                 "--cid", cid, "--server", f"127.0.0.1:{port}", "--num-clients", str(n)],
                cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self.client_processes.append((cid, proc))
        self.btn_start_clients.setEnabled(False)
        self.btn_stop_clients.setEnabled(True)
        self.btn_start_train.setEnabled(True)
        self._log(f"已启动 {n} 个客户端")
        self._refresh_client_table()
        self._refresh_status()

    def stop_all_clients(self):
        for cid, proc in self.client_processes:
            try:
                proc.kill()
            except Exception:
                pass
        self.client_processes.clear()
        self.btn_start_clients.setEnabled(True)
        self.btn_stop_clients.setEnabled(False)
        self.btn_start_train.setEnabled(False)
        self._log("已停止所有客户端")
        self._refresh_client_table()
        self._refresh_status()

    # ================= 训练 =================
    def start_training(self):
        if self.server_process is None or self.server_process.poll() is not None:
            QMessageBox.warning(self, "训练控制", "请先启动服务器")
            return
        if not self.client_processes:
            QMessageBox.warning(self, "训练控制", "请先启动客户端")
            return
        self.is_training = True
        self.current_round = 0
        self._metrics = {}
        self._last_plotted_round = -1
        self._conn_history = []
        # 会话边界：_session_had_history 已在 GUI 打开时确定（旧日志残留）。
        # 这里记录点击时刻的轮次基准：只有解析轮次未超过该基准且新 ROUND 1 未出现时
        # 才保持等待，训练一旦推进（cur > 基准）即正常更新，避免永久卡死
        self._session_click_round = self._peek_round()
        self._session_new_round_seen = False
        self._redraw_charts()   # 立即清空旧曲线，显示"等待开始训练"占位
        self.btn_start_train.setEnabled(False)
        self.btn_start_train.setText("🚀 训练进行中")
        self.btn_pause_train.setEnabled(True)
        self.btn_stop_train.setEnabled(True)
        self.train_status.setText("训练进行中...")
        self._log("开始训练（监控 server.log 真实进度）")
        self._refresh_training()

    def pause_training(self):
        self.is_training = False
        self.btn_start_train.setEnabled(True)
        self.btn_start_train.setText("🚀 继续训练")
        self.btn_pause_train.setEnabled(False)
        self.train_status.setText(f"训练已暂停（第 {self.current_round} 轮）")
        self._log("训练已暂停（服务器仍在运行）")

    def stop_training(self):
        self.is_training = False
        # 手动停止：保留已绘制的曲线，下一次点击"开始训练"时自动清空重新记录
        self.btn_start_train.setEnabled(True)
        self.btn_start_train.setText("🚀 开始训练")
        self.btn_pause_train.setEnabled(False)
        self.btn_stop_train.setEnabled(False)
        self.train_status.setText("训练已停止")
        self._log("训练已停止")

    # ================= 模型管理 =================
    def refresh_model_list(self):
        self.model_list.clear()
        model_dir = self.config.get("model_save_path", "./saved_models")
        if not os.path.exists(model_dir):
            self.model_list.addItem("未找到保存的模型目录")
            return
        try:
            files = sorted([f for f in os.listdir(model_dir)
                            if f.endswith(".h5") and f.startswith("global_model_")],
                           reverse=True)
        except Exception:
            files = []
        if not files:
            self.model_list.addItem("未找到保存的模型")
        else:
            for f in files:
                p = os.path.join(model_dir, f)
                size = os.path.getsize(p) / 1024
                mtime = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m-%d %H:%M")
                self.model_list.addItem(f"{f}  ({size:.0f}KB, {mtime})")
        self._update_model_info()

    def _selected_model(self):
        items = self.model_list.selectedItems()
        if not items:
            return None
        return items[0].text().split("  (")[0]

    def _update_model_info(self):
        name = self._selected_model()
        if not name:
            self.model_info.setText("请从左侧选择模型")
            self.model_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.model_info.setStyleSheet(
                "font-size:15px;color:#a6b0bc;border:1px dashed #d3dae3;border-radius:6px;"
                "padding:24px;background:#f7f9fb;")
            return
        p = os.path.join(self.config.get("model_save_path", "./saved_models"), name)
        if os.path.exists(p):
            size = os.path.getsize(p) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
            self.model_info.setText(
                f"{name}\n\n大小: {size:.0f} KB\n保存时间: {mtime}\n"
                f"路径: {os.path.dirname(p)}")
        else:
            self.model_info.setText("模型文件不存在")
        self.model_info.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.model_info.setStyleSheet(
            "font-size:14px;color:#3d5167;border:none;background:transparent;padding:8px;")
        self.model_list.currentItemChanged.connect(lambda *_: self._update_model_info())

    def load_selected_model(self):
        name = self._selected_model()
        if not name:
            QMessageBox.warning(self, "模型管理", "请先选择一个模型")
            return
        QMessageBox.information(self, "模型管理",
                                f"已选择模型：{name}\n（实际加载/评估在命令行或评估流程中执行）")

    def delete_selected_model(self):
        name = self._selected_model()
        if not name:
            QMessageBox.warning(self, "模型管理", "请先选择一个模型")
            return
        p = os.path.join(self.config.get("model_save_path", "./saved_models"), name)
        if not os.path.exists(p):
            QMessageBox.warning(self, "模型管理", "模型文件未找到")
            return
        if QMessageBox.question(self, "确认删除", f"确定要删除模型 {name} 吗？",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            try:
                os.remove(p)
                self._log(f"已删除模型 {name}")
                self.refresh_model_list()
            except OSError as e:
                QMessageBox.critical(self, "模型管理", f"删除失败: {e}")

    def export_selected_model(self):
        name = self._selected_model()
        if not name:
            QMessageBox.warning(self, "模型导出", "请先选择一个要导出的模型")
            return
        p = os.path.join(self.config.get("model_save_path", "./saved_models"), name)
        if not os.path.exists(p):
            QMessageBox.warning(self, "模型导出", "模型文件未找到")
            return
        dst, _ = QFileDialog.getSaveFileName(self, "导出模型为",
                                             name, "H5 模型 (*.h5)")
        if dst:
            try:
                import shutil
                shutil.copy2(p, dst)
                self._log(f"模型已导出: {dst}")
                QMessageBox.information(self, "模型导出", f"导出成功：{dst}")
            except OSError as e:
                QMessageBox.critical(self, "模型导出", f"导出失败: {e}")

    # ================= 数据解析 =================
    def _server_log_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")

    def _read_client_metric(self, client_id: str):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "client_logs", f"{client_id}.log")
        if not os.path.exists(p):
            return None, None
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                return None, None
            last = json.loads(lines[-1])
            return last.get("samples"), last.get("acc")
        except Exception:
            return None, None

    def _parse_progress(self):
        """流式逐行解析 server.log（O(n) 无正则回溯，日志越大优势越明显）
        返回 (当前轮, {round: {train_loss, train_acc, test_acc}}) + new_session 标志"""
        p = self._server_log_path()
        if not os.path.exists(p):
            return 0, {}, False
        metrics = {}
        cur = 0
        max_seen = 0
        new_session = False
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = re.search(r"\[ROUND (\d+)\]", line)
                    if m:
                        r = int(m.group(1))
                        # 真正的会话回退：曾出现过 >1 的轮次后又出现 ROUND 1
                        # （同一轮内 FitMetrics/EvaluateMetrics 都会带 ROUND N，不误判）
                        if r == 1 and max_seen > 1:
                            metrics.clear()
                            new_session = True
                            max_seen = 1
                        else:
                            max_seen = max(max_seen, r)
                        cur = r
                    if "FitMetrics" in line:
                        r = int(m.group(1)) if m else cur
                        lm = re.search(r"train_loss['\"]?\s*:\s*([\d.eE+-]+)", line)
                        am = re.search(r"train_acc['\"]?\s*:\s*([\d.eE+-]+)", line)
                        d = metrics.setdefault(r, {})
                        if lm:
                            d["train_loss"] = float(lm.group(1))
                        if am:
                            d["train_acc"] = float(am.group(1))
                    elif "EvaluateMetrics" in line:
                        r = int(m.group(1)) if m else cur
                        am = re.search(r"test_acc['\"]?\s*:\s*([\d.eE+-]+)", line)
                        d = metrics.setdefault(r, {})
                        if am:
                            d["test_acc"] = float(am.group(1))
        except Exception:
            pass
        return cur, metrics, new_session

    # ================= 定时刷新 =================
    def _refresh_status(self):
        """3 秒：服务器状态 / 客户端状态 / 底部状态栏 / 性能页指标"""
        server_alive = (self.server_process is not None
                        and self.server_process.poll() is None)

        # 底部状态栏
        if server_alive:
            self.sb_server.setText("服务器运行中")
            self.sb_server.setProperty("status", "ok")
        else:
            self.sb_server.setText("服务器未运行")
            self.sb_server.setProperty("status", "bad")
        self.sb_server.style().unpolish(self.sb_server)
        self.sb_server.style().polish(self.sb_server)
        self.sb_clients.setText(f"客户端: {len(self.client_processes)}")
        self.sb_train.setText(self.train_status.text())
        self.sb_round.setText(f"轮次: {self.current_round}/{self.total_rounds}")

        # 服务器状态页
        if "server" in self._sv_labels:
            self._sv_labels["server"].setText(
                "🟢 运行中" if server_alive else "🔴 未运行")
            self._sv_labels["port"].setText(
                self.port_edit.text() if server_alive else "—")
            self._sv_labels["train"].setText(self.train_status.text())
            self._sv_labels["clients"].setText(str(len(self.client_processes)))
            self._sv_labels["round"].setText(f"{self.current_round}/{self.total_rounds}")
            self._sv_labels["progress"].setText(f"{self.progress.value():.1f}%"
                                                if hasattr(self, "progress") else "—")

        # 真实连接历史（每次刷新记录存活客户端数，供 Connected Clients 曲线）
        alive_clients = sum(1 for _, p in self.client_processes if p.poll() is None)
        hist = getattr(self, "_conn_history", [])
        hist.append(alive_clients)
        if len(hist) > 200:
            del hist[: len(hist) - 200]
        self._conn_history = hist

        self._refresh_client_table()
        self._refresh_perf()

    def _peek_round(self) -> int:
        """轻量读取 server.log 最后出现的轮次号（只读尾部 64KB，不解析指标）"""
        p = self._server_log_path()
        if not os.path.exists(p):
            return 0
        try:
            size = os.path.getsize(p)
            tail_size = min(size, 64 * 1024)
            with open(p, "rb") as f:
                f.seek(size - tail_size)
                chunk = f.read(tail_size).decode("utf-8", errors="ignore")
            rounds = re.findall(r"\[ROUND (\d+)\]", chunk)
            return int(rounds[-1]) if rounds else 0
        except Exception:
            return 0

    def _refresh_client_table(self):
        if not hasattr(self, "client_table"):
            return
        rows = self.client_processes
        met = [self._read_client_metric(cid) for cid, _ in rows]
        sig = tuple((cid, proc.pid, proc.poll() is not None, m[0], m[1])
                    for (cid, proc), m in zip(rows, met))
        if getattr(self, "_client_sig", None) == sig:
            return
        self._client_sig = sig
        self.client_table.setRowCount(len(rows))
        for i, (cid, proc) in enumerate(rows):
            running = proc.poll() is None
            status = "🟢 运行中" if running else "🔴 已停止"
            pid = str(proc.pid) if proc.pid else "—"
            samples, acc = met[i]
            items = [cid, status, pid,
                     str(samples) if samples is not None else "—",
                     f"{acc:.3f}" if acc is not None else "—"]
            for j, v in enumerate(items):
                item = QTableWidgetItem(v)
                if j == 1:
                    item.setForeground(
                        __import__("PySide6").QtGui.QColor(C_SUCCESS if running else C_DANGER))
                self.client_table.setItem(i, j, item)

    def _refresh_training(self):
        """2 秒：轻量探测轮次；仅新轮次时解析指标并重绘（防止主线程被拖垮）"""
        if not self.is_training:
            return
        if self.server_process is not None and self.server_process.poll() is not None:
            self.is_training = False
            self.train_status.setText("训练: 服务器已退出")
            return
        cur, metrics, new_session = self._parse_progress()
        if new_session:
            self._session_new_round_seen = True
        # 旧日志残留、新训练尚未开始（新 ROUND 1 未出现）且解析轮次未超过点击时基准：
        # 保持等待，不更新进度/曲线/完成状态，避免"点一次开始训练就 100% 已完成"的假象。
        # 训练一旦真实推进（cur 超过点击基准），立即正常更新。
        if (self._session_had_history and not self._session_new_round_seen
                and cur <= getattr(self, "_session_click_round", -1)):
            return
        self.current_round = cur

        if cur > 0:
            pct = min(100.0, cur / max(1, self.total_rounds) * 100)
            self.progress.setValue(int(pct))
            self.monitor_progress.setValue(int(pct))
            self.monitor_round_label.setText(f"轮次: {cur}/{self.total_rounds}")
            self.train_status.setText(f"训练中 (第{cur}/{self.total_rounds}轮)")
        else:
            self.train_status.setText("训练中 (等待第一轮完成...)")

        # 仅在新轮次出现时解析指标并重绘（每轮一次，而非每 2 秒一次）
        if cur != getattr(self, "_last_plotted_round", -1):
            self._last_plotted_round = cur
            if cur > 0:
                self._metrics = metrics
                self._redraw_charts()

        if cur >= self.total_rounds:
            self.is_training = False
            # 训练完成：保留曲线供查看，下一次点击"开始训练"时自动清空重新记录
            self.train_status.setText("训练: 已完成")
            self.btn_start_train.setEnabled(True)
            self.btn_start_train.setText("🚀 开始训练")
            self.btn_pause_train.setEnabled(False)
            self.btn_stop_train.setEnabled(False)
            self._log("联邦学习训练已完成")

    def _redraw_charts(self):
        """重绘 4 个图表：有指标画真实曲线；无指标清空并显示占位（绝不残留上一轮数据）"""
        try:
            for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
                ax.clear()
                ax.grid(True, alpha=0.3)
            if not self._metrics:
                # 未开始训练 / 尚未产生指标：显示占位，绝不残留上一轮曲线
                titles = ["Accuracy Trend", "Loss Trend",
                          "Connected Clients", "Sample Distribution (Non-IID)"]
                for ax, t in zip((self.ax1, self.ax2, self.ax3, self.ax4), titles):
                    ax.set_title(t)
                    ax.text(0.5, 0.5, "等待开始训练…\n启动服务器并开始训练后自动更新",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=10, color="#8a97a8")
                self.monitor_acc_label.setText("test_acc: —")
                self.canvas.draw()
                return

            rounds = sorted(self._metrics.keys())
            train_acc = [self._metrics[r].get("train_acc", 0) for r in rounds]
            test_acc = [self._metrics[r].get("test_acc", 0) for r in rounds]
            train_loss = [self._metrics[r].get("train_loss", 0) for r in rounds]
            latest = self._metrics[rounds[-1]]
            self.monitor_acc_label.setText(f"test_acc: {latest.get('test_acc', 0):.3f}")

            self.ax1.set_title("Accuracy Trend")
            self.ax1.set_xlabel("训练轮数")
            self.ax1.set_ylabel("准确率")
            self.ax1.plot(rounds, train_acc, color=C_PRIMARY, linewidth=2, label="train_acc")
            if any(v for v in test_acc):
                self.ax1.plot(rounds, test_acc, color=C_SUCCESS, linestyle="--",
                              linewidth=2, label="test_acc")
            self.ax1.legend(fontsize=8)
            self.ax2.set_title("Loss Trend")
            self.ax2.set_xlabel("训练轮数")
            self.ax2.set_ylabel("损失值")
            self.ax2.plot(rounds, train_loss, color="#e67e22", linewidth=2)
            # Connected Clients：真实连接历史（每次刷新记录存活客户端数）
            hist = getattr(self, "_conn_history", [])
            self.ax3.set_title("Connected Clients")
            self.ax3.set_xlabel("时间（秒）")
            self.ax3.set_ylabel("客户端数量")
            if hist:
                self.ax3.plot(range(len(hist)), hist, color=C_SUCCESS,
                              linewidth=2, marker="o", markersize=3)
            self.ax4.set_title("Sample Distribution (Non-IID)")
            self.ax4.set_xlabel("客户端")
            self.ax4.set_ylabel("样本数量")
            cids = [f"C{i + 1}" for i in range(len(self.client_processes))]
            samples = [1000] * len(cids)
            bars = self.ax4.bar(cids, samples, color=C_PRIMARY, alpha=0.75)
            for bar in bars:
                h = bar.get_height()
                self.ax4.text(bar.get_x() + bar.get_width() / 2., h,
                              f"{int(h)}", ha="center", va="bottom", fontsize=8)
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"重绘图表失败: {e}")

    def _refresh_log(self):
        """3 秒：server.log 增量读取 → 控制台 + 日志页（避免全量读取阻塞主线程）"""
        try:
            p = self._server_log_path()
            if not os.path.exists(p):
                self._log_pos = 0
                return
            size = os.path.getsize(p)
            pos = getattr(self, "_log_pos", 0)
            if size < pos:      # 日志被清理/重建
                pos = 0
            if size == pos:
                return
            with open(p, "rb") as f:
                f.seek(pos)
                data = f.read(size - pos).decode("utf-8", errors="ignore")
            self._log_pos = size
            if not data.strip():
                return
            lines = data.splitlines()[-80:]
            for line in lines:
                if hasattr(self, "console"):
                    self.console.insertPlainText(line + "\n")
                if hasattr(self, "log_view"):
                    self.log_view.insertPlainText(line + "\n")
            for w in (self.console, self.log_view):
                w.setMaximumBlockCount(2500)
        except Exception:
            pass

    def closeEvent(self, event):
        """退出时清理子进程"""
        try:
            if self.server_process is not None:
                self.server_process.kill()
            for cid, proc in self.client_processes:
                try:
                    proc.kill()
                except Exception:
                    pass
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    lock = _acquire_single_instance_lock()
    if lock is None:
        QMessageBox.warning(None, "已有实例在运行",
                            "联邦学习系统 GUI 已在运行。\n\n"
                            "重复启动会导致端口冲突与训练卡死，"
                            "请使用已打开的窗口。")
        return 1

    win = FederatedGUI()
    win.show()
    code = app.exec()
    try:
        lock.close()
    except Exception:
        pass
    return code


if __name__ == "__main__":
    sys.exit(main())
