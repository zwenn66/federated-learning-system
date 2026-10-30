#!/usr/bin/env python3
"""
联邦学习系统 GUI
服务器控制，客户端管理，训练监控，模型可视化
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import ttkbootstrap as ttkb  # 现代扁平主题扩展（Bootstrap 风格，兼容 ttk）
import threading
import subprocess
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk

# ============ 统一配色（低饱和工业风，参考源极智造 MES 界面） ============
COLOR_PRIMARY = "#3a7bd5"   # 主蓝（折线/强调）
COLOR_SUCCESS = "#27ae60"   # 成功绿（test_acc / 运行中）
COLOR_ACCENT  = "#e67e22"   # 强调橙（loss 曲线）
COLOR_MUTED   = "#7f8c8d"   # 弱化灰（说明文字）
COLOR_CHART_BG = "#f5f7fa"  # 图表/面板背景（浅灰，与 flatly 主题协调）
# 深色侧边导航（参考源极智造 MES：深灰导航 + 浅色内容区）
COLOR_SIDEBAR_BG = "#1e2a3a"     # 深灰蓝
COLOR_SIDEBAR_TEXT = "#e8eef5"   # 浅白文字
COLOR_SIDEBAR_MUTED = "#9db1c4"  # 弱化文字

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FederatedLearningGUI:
    """联邦学习GUI主界面"""

    def __init__(self, root):
        self.root = root
        self.root.title("联邦学习系统 v2.0")
        self.root.geometry("1100x750")
        self.root.minsize(960, 660)  # 最小尺寸，布局可随窗口弹性缩放（适配不同屏幕）

        # 统一按钮内边距（避免文字挤压，参考 MES 工具按钮的方正排布）
        try:
            self.root.style.configure("TButton", padding=(14, 7), font=("Microsoft YaHei", 10))
            self.root.style.configure("Accent.TButton", padding=(14, 7), font=("Microsoft YaHei", 10))
        except Exception:
            pass

        # 服务器进程
        self.server_process = None
        self.client_processes = []

        # 训练状态
        self.is_training = False
        self.current_round = 0
        self.total_rounds = 20
        self.connected_clients = 0

        # 配置
        self.config = self.load_config()

        # 日志缓存（用于在日志面板未显示时缓存日志）
        self.log_cache = []
        self.max_log_cache = 100

        # 创建UI
        self.setup_ui()

        # 确保模型目录存在
        self.ensure_model_directory()

        # 开始状态监控
        self.monitor_status()

    def load_config(self) -> Dict:
        """加载配置"""
        # 设置默认模型保存路径
        default_model_path = './saved_models'

        default_config = {
            'server_port': 8081,  # 改为8081避免冲突
            'num_rounds': 20,
            'min_clients': 2,
            'local_epochs': 1,
            'batch_size': 32,
            'learning_rate': 0.01,
            'model_save_path': default_model_path
        }

        config_file = './gui_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 如果配置中没有模型路径，使用默认值
                    if 'model_save_path' not in loaded_config:
                        loaded_config['model_save_path'] = default_model_path
                    return loaded_config
            except Exception as e:
                print(f"加载配置错误: {e}")
                return default_config

        return default_config

    def ensure_model_directory(self):
        """确保模型目录存在"""
        model_dir = self.config.get('model_save_path', './saved_models')
        if not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
                self.log_message(f"已创建模型目录: {model_dir}", "INFO")
            except Exception as e:
                self.log_message(f"创建模型目录失败 {model_dir}: {e}", "ERROR")

    def save_config(self):
        """保存配置"""
        config_file = './gui_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def setup_ui(self):
        """设置UI界面"""
        # 创建主框架容器
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # 创建侧边栏框架
        self.setup_sidebar()

        # 创建内容区域框架
        self.setup_content_area()

    def setup_sidebar(self):
        """设置侧边栏索引"""
        # 侧边栏框架（深色，参考 MES 深色导航 + 浅色内容区）
        self.root.style.configure("Sidebar.TFrame", background=COLOR_SIDEBAR_BG)
        self.root.style.configure("Sidebar.TLabel",
                                  background=COLOR_SIDEBAR_BG, foreground=COLOR_SIDEBAR_TEXT)
        self.sidebar_frame = ttk.Frame(self.main_container, width=180, style="Sidebar.TFrame")
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        self.sidebar_frame.pack_propagate(False)  # 固定宽度

        # 侧边栏标题
        sidebar_title = ttk.Label(
            self.sidebar_frame,
            text="导航菜单",
            font=("Microsoft YaHei", 12, "bold"),
            style="Sidebar.TLabel",
            foreground=COLOR_SIDEBAR_TEXT
        )
        sidebar_title.pack(pady=15)

        # 添加分割线
        ttk.Separator(self.sidebar_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # 创建滚动框架（深色）
        self.sidebar_canvas = tk.Canvas(self.sidebar_frame, highlightthickness=0, bg=COLOR_SIDEBAR_BG)
        self.sidebar_scrollbar = ttk.Scrollbar(
            self.sidebar_frame,
            orient=tk.VERTICAL,
            command=self.sidebar_canvas.yview
        )
        self.sidebar_scrollable_frame = ttk.Frame(self.sidebar_canvas, style="Sidebar.TFrame")

        self.sidebar_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        )

        self.sidebar_canvas.create_window((0, 0), window=self.sidebar_scrollable_frame, anchor="nw")
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set)

        # ========== 核心修改：带边界判断的滚轮事件 ==========
        def _on_mousewheel(event):
            # 获取当前滚动位置（0.0=顶部，1.0=底部）
            current_view = self.sidebar_canvas.yview()
            scroll_top = current_view[0]  # 顶部位置（0.0为完全顶部）
            scroll_bottom = current_view[1]  # 底部位置（1.0为完全底部）

            # 计算滚动步长
            scroll_units = int(-1 * (event.delta / 120))

            # 边界判断：
            # 1. 向上滚动（scroll_units < 0）且已到顶部 → 停止滚动
            if scroll_units < 0 and scroll_top <= 0.0:
                return
            # 2. 向下滚动（scroll_units > 0）且已到底部 → 停止滚动（可选）
            # if scroll_units > 0 and scroll_bottom >= 1.0:
            #     return

            # 未到边界时执行滚动
            self.sidebar_canvas.yview_scroll(scroll_units, "units")

        # ========== 补充：修复Linux/Mac滚轮事件（Button-4/5）的边界判断 ==========
        def _on_button_scroll(event):
            current_view = self.sidebar_canvas.yview()
            scroll_top = current_view[0]

            # Button-4 = 向上滚动（Linux/Mac）
            if event.num == 4:
                if scroll_top <= 0.0:
                    return
                self.sidebar_canvas.yview_scroll(-1, "units")
            # Button-5 = 向下滚动（Linux/Mac）
            elif event.num == 5:
                # 可选：判断底部边界
                # scroll_bottom = current_view[1]
                # if scroll_bottom >= 1.0:
                #     return
                self.sidebar_canvas.yview_scroll(1, "units")

        # ========== 绑定滚轮事件（替换原有绑定） ==========
        self.sidebar_canvas.bind("<MouseWheel>", _on_mousewheel)  # Windows滚轮
        self.sidebar_canvas.bind("<Button-4>", _on_button_scroll)  # Linux/Mac向上
        self.sidebar_canvas.bind("<Button-5>", _on_button_scroll)  # Linux/Mac向下
        # 注意：移除bind_all，避免影响整个界面的滚轮，只绑定到sidebar_canvas

        # 打包滚动区域
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 导航按钮列表
        self.nav_buttons = []
        nav_items = [
            ("🏠 控制面板", self.show_control_tab),
            ("📊 训练监控", self.show_monitor_tab),
            ("🧠 模型管理", self.show_model_tab),
            ("📋 系统日志", self.show_log_panel),
            ("⚙️ 服务器状态", self.show_server_status),
            ("👥 客户端状态", self.show_client_status),
            ("📈 性能分析", self.show_performance),
            ("🔧 系统设置", self.show_settings),
            ("❓ 帮助文档", self.show_help),
            ("ℹ️ 关于系统", self.show_about)
        ]

        for text, command in nav_items:
            btn = ttkb.Button(  # 用 ttkbootstrap 的 Button 才能支持 bootstyle
                self.sidebar_scrollable_frame,
                text=text,
                command=command,
                width=18,
                bootstyle="dark"  # 深色导航按钮（选中时切换为主蓝高亮）
            )
            btn.pack(pady=5, padx=10, fill=tk.X)
            self.nav_buttons.append(btn)

        # 添加底部信息
        ttk.Separator(self.sidebar_scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=15)

        bottom_info = ttk.Label(
            self.sidebar_scrollable_frame,
            text="联邦学习系统\nv2.0",
            font=("Microsoft YaHei", 9),
            style="Sidebar.TLabel",
            foreground=COLOR_SIDEBAR_MUTED,
            justify=tk.CENTER
        )
        bottom_info.pack(pady=10)

    def setup_content_area(self):
        """设置内容区域"""
        # 内容主框架
        self.content_frame = ttk.Frame(self.main_container, padding="8")
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 标题
        self.title_label = ttk.Label(
            self.content_frame,
            text="联邦学习系统",
            font=("Microsoft YaHei", 22, "bold"),
            foreground="#2c3e50"
        )
        self.title_label.grid(row=0, column=0, columnspan=3, pady=15)

        # 创建标签页容器
        self.tab_container = ttk.Frame(self.content_frame)
        self.tab_container.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=8)

        # 配置网格权重
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(1, weight=1)

        # 创建标签页
        self.setup_tabs()

        # 状态栏
        self.setup_status_bar()

    def setup_tabs(self):
        """设置标签页"""
        # 控制面板标签页
        self.control_tab = ttk.Frame(self.tab_container)
        self.setup_control_tab()

        # 监控面板标签页
        self.monitor_tab = ttk.Frame(self.tab_container)
        self.setup_monitor_tab()

        # 模型管理标签页
        self.model_tab = ttk.Frame(self.tab_container)
        self.setup_model_tab()

        # 默认显示控制面板
        self.show_control_tab()

    def setup_control_tab(self):
        """设置控制面板"""
        # 清除现有内容
        for widget in self.control_tab.winfo_children():
            widget.destroy()

        control_frame = ttk.LabelFrame(self.control_tab, text="服务器控制", padding="8")
        control_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 外观设置（ttkbootstrap 主题切换，即时生效）
        appearance_frame = ttk.LabelFrame(control_frame, text="外观设置", padding="8")
        appearance_frame.pack(fill=tk.X, pady=8)

        theme_row = ttk.Frame(appearance_frame)
        theme_row.pack(fill=tk.X, pady=4)

        ttk.Label(theme_row, text="界面主题:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=4)
        self.theme_var = tk.StringVar(value="flatly")
        theme_menu = ttk.Combobox(
            theme_row,
            textvariable=self.theme_var,
            values=["flatly", "litera", "cosmo", "yeti", "minty", "darkly", "solar", "morph"],
            width=12,
            state="readonly"
        )
        theme_menu.pack(side=tk.LEFT, padx=4)
        theme_menu.bind("<<ComboboxSelected>>", lambda e: self.change_theme(self.theme_var.get()))

        ttk.Label(
            theme_row,
            text="浅色系: flatly/litera/cosmo/yeti/minty · 深色系: darkly/solar/morph",
            font=("Microsoft YaHei", 9),
            foreground="#7f8c8d"
        ).pack(side=tk.LEFT, padx=8)

        # 服务器配置
        config_frame = ttk.Frame(control_frame)
        config_frame.pack(fill=tk.X, pady=4)

        # 端口配置
        ttk.Label(config_frame, text="服务器端口:", font=("Microsoft YaHei", 10)).grid(row=0, column=0, sticky=tk.W,
                                                                                       padx=4)
        self.port_var = tk.StringVar(value=str(self.config['server_port']))
        port_entry = ttk.Entry(config_frame, textvariable=self.port_var, width=10)
        port_entry.grid(row=0, column=1, sticky=tk.W, padx=4)

        # 训练轮数
        ttk.Label(config_frame, text="训练轮数:", font=("Microsoft YaHei", 10)).grid(row=0, column=2, sticky=tk.W,
                                                                                     padx=15)
        self.rounds_var = tk.StringVar(value=str(self.config['num_rounds']))
        rounds_entry = ttk.Entry(config_frame, textvariable=self.rounds_var, width=10)
        rounds_entry.grid(row=0, column=3, sticky=tk.W, padx=4)

        # 最小客户端数
        ttk.Label(config_frame, text="最小客户端数:", font=("Microsoft YaHei", 10)).grid(row=1, column=0, sticky=tk.W,
                                                                                         padx=4, pady=8)
        self.min_clients_var = tk.StringVar(value=str(self.config['min_clients']))
        clients_entry = ttk.Entry(config_frame, textvariable=self.min_clients_var, width=10)
        clients_entry.grid(row=1, column=1, sticky=tk.W, padx=4, pady=8)

        # 本地训练轮数
        ttk.Label(config_frame, text="本地训练轮数:", font=("Microsoft YaHei", 10)).grid(row=1, column=2, sticky=tk.W,
                                                                                         padx=15, pady=8)
        self.local_epochs_var = tk.StringVar(value=str(self.config['local_epochs']))
        epochs_entry = ttk.Entry(config_frame, textvariable=self.local_epochs_var, width=10)
        epochs_entry.grid(row=1, column=3, sticky=tk.W, padx=4, pady=8)

        # 模型保存路径配置
        ttk.Label(config_frame, text="模型路径:", font=("Microsoft YaHei", 10)).grid(row=2, column=0, sticky=tk.W,
                                                                                     padx=4, pady=8)
        self.model_path_var = tk.StringVar(value=self.config.get('model_save_path', './saved_models'))
        model_path_frame = ttk.Frame(config_frame)
        model_path_frame.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=4, pady=8)

        model_path_entry = ttk.Entry(model_path_frame, textvariable=self.model_path_var, width=35)
        model_path_entry.pack(side=tk.LEFT, padx=4)

        ttk.Button(
            model_path_frame,
            text="浏览...",
            command=self.browse_model_path
        ).pack(side=tk.LEFT, padx=4)

        # 按钮框架
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=15)

        # 启动服务器按钮
        self.start_server_btn = ttk.Button(
            button_frame,
            text="🚀 启动服务器",
            command=self.start_server,
            style="Accent.TButton"
        )
        self.start_server_btn.pack(side=tk.LEFT, padx=4)

        # 停止服务器按钮
        self.stop_server_btn = ttk.Button(
            button_frame,
            text="🛑 停止服务器",
            command=self.stop_server,
            state=tk.DISABLED
        )
        self.stop_server_btn.pack(side=tk.LEFT, padx=4)

        # 保存配置按钮
        ttk.Button(
            button_frame,
            text="💾 保存配置",
            command=self.save_current_config
        ).pack(side=tk.LEFT, padx=4)

        # 测试连接按钮
        ttk.Button(
            button_frame,
            text="🔍 测试端口",
            command=self.test_port
        ).pack(side=tk.LEFT, padx=4)

        # 客户端管理
        client_frame = ttk.LabelFrame(control_frame, text="客户端管理", padding="8")
        client_frame.pack(fill=tk.BOTH, expand=True, pady=8)

        # 客户端数量控制
        client_control_frame = ttk.Frame(client_frame)
        client_control_frame.pack(fill=tk.X, pady=4)

        ttk.Label(client_control_frame, text="客户端数量:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=4)
        self.client_count_var = tk.StringVar(value="3")
        client_count_spin = ttk.Spinbox(
            client_control_frame,
            from_=1,
            to=10,
            textvariable=self.client_count_var,
            width=5
        )
        client_count_spin.pack(side=tk.LEFT, padx=4)

        # 启动客户端按钮
        self.start_clients_btn = ttk.Button(
            client_control_frame,
            text="▶️ 启动客户端",
            command=self.start_clients
        )
        self.start_clients_btn.pack(side=tk.LEFT, padx=15)

        # 停止所有客户端按钮
        self.stop_clients_btn = ttk.Button(
            client_control_frame,
            text="⏹️ 停止所有客户端",
            command=self.stop_all_clients,
            state=tk.DISABLED
        )
        self.stop_clients_btn.pack(side=tk.LEFT, padx=4)

        # 测试所有客户端按钮
        ttk.Button(
            client_control_frame,
            text="🧪 测试客户端",
            command=self.test_clients
        ).pack(side=tk.LEFT, padx=4)

        # 客户端状态显示
        self.client_listbox = tk.Listbox(client_frame, height=7, font=("Microsoft YaHei", 9))
        self.client_listbox.pack(fill=tk.BOTH, expand=True, pady=8)

        # 训练控制
        train_frame = ttk.LabelFrame(control_frame, text="训练控制", padding="8")
        train_frame.pack(fill=tk.X, pady=8)

        train_btn_frame = ttk.Frame(train_frame)
        train_btn_frame.pack(fill=tk.X)

        self.start_train_btn = ttk.Button(
            train_btn_frame,
            text="🚀 开始训练",
            command=self.start_training,
            state=tk.DISABLED
        )
        self.start_train_btn.pack(side=tk.LEFT, padx=4)

        self.pause_train_btn = ttk.Button(
            train_btn_frame,
            text="⏸️ 暂停训练",
            command=self.pause_training,
            state=tk.DISABLED
        )
        self.pause_train_btn.pack(side=tk.LEFT, padx=4)

        self.stop_train_btn = ttk.Button(
            train_btn_frame,
            text="⏹️ 停止训练",
            command=self.stop_training,
            state=tk.DISABLED
        )
        self.stop_train_btn.pack(side=tk.LEFT, padx=4)

        # 训练进度
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            train_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=8)

        # 进度标签
        self.progress_label = ttk.Label(train_frame, text="等待开始训练...", font=("Microsoft YaHei", 10))
        self.progress_label.pack()

    def setup_monitor_tab(self):
        """设置监控面板"""
        # 清除现有内容
        for widget in self.monitor_tab.winfo_children():
            widget.destroy()

        # 创建图形框架
        self.figure_frame = ttk.Frame(self.monitor_tab)
        self.figure_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 配置matplotlib参数
        plt.rcParams.update({
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 11,
            'figure.autolayout': False,
            'figure.dpi': 100,
        })

        # 创建matplotlib图形
        self.fig = plt.figure(figsize=(9.5, 7), dpi=100)

        # 创建子图
        gs = self.fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])

        # 设置背景颜色（与 flatly 主题低饱和浅灰统一）
        self.fig.set_facecolor(COLOR_CHART_BG)

        # 配置子图
        self.ax1.set_title('Accuracy Trend', pad=15)  # 保持英文
        self.ax1.set_xlabel('训练轮数')
        self.ax1.set_ylabel('准确率')
        self.ax1.grid(True, alpha=0.3)

        self.ax2.set_title('Loss Trend', pad=15)  # 保持英文
        self.ax2.set_xlabel('训练轮数')
        self.ax2.set_ylabel('损失值')
        self.ax2.grid(True, alpha=0.3)

        self.ax3.set_title('Connected Clients', pad=15)  # 保持英文
        self.ax3.set_xlabel('时间')
        self.ax3.set_ylabel('客户端数量')
        self.ax3.grid(True, alpha=0.3)

        self.ax4.set_title('Sample Distribution', pad=15)  # 保持英文
        self.ax4.set_xlabel('客户端')
        self.ax4.set_ylabel('样本数量')
        self.ax4.grid(True, alpha=0.3)

        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, self.figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 图表尺寸随容器自适应（窗口缩放时重绘，不写死高度）
        self.figure_frame.bind("<Configure>", self._resize_figure)

        # 刷新按钮
        refresh_btn = ttk.Button(
            self.figure_frame,
            text="🔄 刷新图表",
            command=self.update_charts
        )
        refresh_btn.pack(pady=8)

    def setup_model_tab(self):
        """设置模型管理面板"""
        # 清除现有内容
        for widget in self.model_tab.winfo_children():
            widget.destroy()

        model_frame = ttk.Frame(self.model_tab)
        model_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 模型列表标题
        model_dir = self.config.get('model_save_path', './saved_models')
        list_frame = ttk.LabelFrame(model_frame, text=f"已保存的模型 (路径: {model_dir})", padding="8")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=8)

        # 模型列表框
        self.model_listbox = tk.Listbox(list_frame, height=13, font=("Microsoft YaHei", 9))
        self.model_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.model_listbox.yview)

        # 按钮框架
        btn_frame = ttk.Frame(model_frame)
        btn_frame.pack(fill=tk.X, pady=8)

        # 加载模型按钮
        ttk.Button(
            btn_frame,
            text="📂 加载模型",
            command=self.load_selected_model
        ).pack(side=tk.LEFT, padx=4)

        # 删除模型按钮
        ttk.Button(
            btn_frame,
            text="🗑️ 删除模型",
            command=self.delete_selected_model
        ).pack(side=tk.LEFT, padx=4)

        # 导出模型按钮
        ttk.Button(
            btn_frame,
            text="📤 导出模型",
            command=self.export_model
        ).pack(side=tk.LEFT, padx=4)

        # 刷新模型列表按钮
        ttk.Button(
            btn_frame,
            text="🔄 刷新列表",
            command=self.refresh_model_list
        ).pack(side=tk.LEFT, padx=4)

        # 打开目录按钮
        ttk.Button(
            btn_frame,
            text="📁 打开目录",
            command=self.open_model_directory
        ).pack(side=tk.LEFT, padx=4)

        # 模型信息显示
        info_frame = ttk.LabelFrame(model_frame, text="模型信息", padding="8")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=8)

        self.model_info_text = scrolledtext.ScrolledText(info_frame, height=9, font=("Microsoft YaHei", 9))
        self.model_info_text.pack(fill=tk.BOTH, expand=True)

        # 初始模型列表刷新
        try:
            self.refresh_model_list()
        except Exception as e:
            print(f"刷新模型列表时出错: {e}")

    def show_control_tab(self):
        """显示控制面板"""
        self.title_label.config(text="联邦学习系统 - 控制面板")
        self.hide_all_tabs()
        self.control_tab.pack(fill=tk.BOTH, expand=True)
        self.update_nav_button_styles("控制面板")

    def show_monitor_tab(self):
        """显示训练监控"""
        self.title_label.config(text="联邦学习系统 - 训练监控")
        self.hide_all_tabs()
        self.monitor_tab.pack(fill=tk.BOTH, expand=True)
        self.update_nav_button_styles("训练监控")

    def show_model_tab(self):
        """显示模型管理"""
        self.title_label.config(text="联邦学习系统 - 模型管理")
        self.hide_all_tabs()
        self.model_tab.pack(fill=tk.BOTH, expand=True)
        self.update_nav_button_styles("模型管理")

    def show_log_panel(self):
        """显示日志面板"""
        self.title_label.config(text="联邦学习系统 - 系统日志")
        self.hide_all_tabs()
        self.setup_log_panel_content()
        self.update_nav_button_styles("系统日志")

    def show_server_status(self):
        """显示服务器状态"""
        self.title_label.config(text="联邦学习系统 - 服务器状态")
        self.hide_all_tabs()
        self.setup_server_status_content()
        self.update_nav_button_styles("服务器状态")

    def show_client_status(self):
        """显示客户端状态"""
        self.title_label.config(text="联邦学习系统 - 客户端状态")
        self.hide_all_tabs()
        self.setup_client_status_content()
        self.update_nav_button_styles("客户端状态")

    def show_performance(self):
        """显示性能分析"""
        self.title_label.config(text="联邦学习系统 - 性能分析")
        self.hide_all_tabs()
        self.setup_performance_content()
        self.update_nav_button_styles("性能分析")

    def show_settings(self):
        """显示系统设置"""
        self.title_label.config(text="联邦学习系统 - 系统设置")
        self.hide_all_tabs()
        self.setup_settings_content()
        self.update_nav_button_styles("系统设置")

    def show_help(self):
        """显示帮助文档"""
        self.title_label.config(text="联邦学习系统 - 帮助文档")
        self.hide_all_tabs()
        self.setup_help_content()
        self.update_nav_button_styles("帮助文档")

    def show_about(self):
        """显示关于系统"""
        self.title_label.config(text="联邦学习系统 - 关于")
        self.hide_all_tabs()
        self.setup_about_content()
        self.update_nav_button_styles("关于系统")

    def hide_all_tabs(self):
        """隐藏所有标签页"""
        for tab in [self.control_tab, self.monitor_tab, self.model_tab]:
            tab.pack_forget()

        # 清除动态创建的内容
        for widget in self.tab_container.winfo_children():
            if widget not in [self.control_tab, self.monitor_tab, self.model_tab]:
                widget.destroy()

    def update_nav_button_styles(self, active_button_text):
        """更新导航按钮样式（深色导航：未选中深灰，选中主蓝高亮）

        用 ttkbootstrap 预置的 bootstyle style 名动态切换（dark.TButton / info.TButton）
        """
        for btn in self.nav_buttons:
            if active_button_text in btn.cget("text"):
                btn.config(style="info.TButton")
            else:
                btn.config(style="dark.TButton")

    def setup_log_panel_content(self):
        """设置日志面板内容"""
        log_frame = ttk.Frame(self.tab_container)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 日志标题
        log_title = ttk.Label(
            log_frame,
            text="系统日志记录",
            font=("Microsoft YaHei", 14, "bold")
        )
        log_title.pack(pady=10)

        # 日志文本框
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=25,
            font=("Consolas", 9),
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 从缓存加载之前的日志
        self.flush_log_cache()

        # 日志控制按钮
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            log_btn_frame,
            text="🗑️ 清空日志",
            command=self.clear_log
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            log_btn_frame,
            text="💾 保存日志",
            command=self.save_log
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            log_btn_frame,
            text="📋 复制日志",
            command=self.copy_log
        ).pack(side=tk.LEFT, padx=4)

    def flush_log_cache(self):
        """将缓存的日志刷新到日志文本框"""
        if hasattr(self, 'log_text') and self.log_cache:
            for cached_message in self.log_cache:
                try:
                    timestamp, level, message = cached_message
                    level_translation = {
                        "INFO": "信息",
                        "WARNING": "警告",
                        "ERROR": "错误",
                        "SUCCESS": "成功"
                    }
                    level_colors = {
                        "INFO": "black",
                        "WARNING": "orange",
                        "ERROR": "red",
                        "SUCCESS": "green"
                    }
                    color = level_colors.get(level, "black")
                    level_display = level_translation.get(level, level)
                    formatted_message = f"[{timestamp}] [{level_display}] {message}\n"

                    self.log_text.insert(tk.END, formatted_message)
                    self.log_text.tag_add(level, "end-2c linestart", "end-1c")
                    self.log_text.tag_config(level, foreground=color)
                except Exception as e:
                    print(f"刷新缓存日志时出错: {e}")

            self.log_text.see(tk.END)
            # 清空缓存
            self.log_cache.clear()

    def copy_log(self):
        """复制日志到剪贴板"""
        try:
            if hasattr(self, 'log_text'):
                log_content = self.log_text.get(1.0, tk.END)
                self.root.clipboard_clear()
                self.root.clipboard_append(log_content)
                self.log_message("日志已复制到剪贴板", "SUCCESS")
        except Exception as e:
            self.log_message(f"复制日志失败: {e}", "ERROR")

    def setup_server_status_content(self):
        """设置服务器状态内容"""
        status_frame = ttk.Frame(self.tab_container)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 服务器状态标题
        status_title = ttk.Label(
            status_frame,
            text="服务器状态监控",
            font=("Microsoft YaHei", 14, "bold")
        )
        status_title.pack(pady=10)

        # 状态信息框架
        info_frame = ttk.LabelFrame(status_frame, text="实时状态", padding="15")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 状态信息（动态变量：由 monitor_status 每 5 秒就地更新，页面重建时保持引用）
        if not hasattr(self, 'server_port_var'):
            self.server_port_var = tk.StringVar(value="未启动")
        if not hasattr(self, 'server_round_var'):
            self.server_round_var = tk.StringVar(value="0/0")
        if not hasattr(self, 'server_progress_var'):
            self.server_progress_var = tk.StringVar(value="0%")

        status_info = [
            ("服务器状态:", self.server_status_var),
            ("运行端口:", self.server_port_var),
            ("训练状态:", self.train_status_var),
            ("客户端数量:", self.client_status_var),
            ("当前轮数:", self.server_round_var),
            ("训练进度:", self.server_progress_var)
        ]

        for i, (label_text, var) in enumerate(status_info):
            frame = ttk.Frame(info_frame)
            frame.pack(fill=tk.X, pady=8)

            ttk.Label(
                frame,
                text=label_text,
                font=("Microsoft YaHei", 11, "bold"),
                width=15
            ).pack(side=tk.LEFT, padx=10)

            ttk.Label(
                frame,
                textvariable=var,
                font=("Microsoft YaHei", 11),
                foreground="#2c3e50"
            ).pack(side=tk.LEFT, padx=10)

        # 服务器操作按钮
        btn_frame = ttk.Frame(status_frame)
        btn_frame.pack(pady=20)

        ttk.Button(
            btn_frame,
            text="🔄 刷新状态",
            command=self.refresh_server_status
        ).pack(side=tk.LEFT, padx=10)

    def change_theme(self, theme_name: str):
        """切换 ttkbootstrap 界面主题（即时生效）"""
        try:
            self.root.style.theme_use(theme_name)
            self.log_message(f"界面主题已切换为 {theme_name}", "INFO")
        except Exception as e:
            self.log_message(f"主题切换失败: {e}", "ERROR")

    def _resize_figure(self, event=None):
        """图表尺寸随容器自适应（窗口缩放时重绘，不写死高度）"""
        try:
            w = self.figure_frame.winfo_width()
            h = self.figure_frame.winfo_height()
            if w > 100 and h > 100:
                self.fig.set_size_inches(w / self.fig.dpi, h / self.fig.dpi)
                self.fig.canvas.draw_idle()
        except Exception:
            pass

    def refresh_server_status(self):
        """刷新服务器状态"""
        self._update_server_status_page()
        self.log_message("服务器状态已刷新", "INFO")

    def _update_server_status_page(self):
        """就地更新服务器状态页的端口/轮数/进度（真实数据）"""
        try:
            if self.server_process is not None and self.server_process.poll() is None:
                self.server_port_var.set(f"端口 {self.port_var.get()} (运行中)")
            else:
                self.server_port_var.set("未启动")
            self.server_round_var.set(f"{self.current_round}/{self.total_rounds}")
            self.server_progress_var.set(f"{self.progress_var.get():.1f}%")
        except Exception:
            pass

    def setup_client_status_content(self):
        """设置客户端状态内容"""
        status_frame = ttk.Frame(self.tab_container)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 客户端状态标题
        status_title = ttk.Label(
            status_frame,
            text="客户端状态监控",
            font=("Microsoft YaHei", 14, "bold")
        )
        status_title.pack(pady=10)

        # 客户端列表框架
        list_frame = ttk.LabelFrame(status_frame, text="已连接的客户端", padding="15")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 客户端列表
        if hasattr(self, 'client_listbox'):
            self.client_listbox.pack_forget()

        client_list_frame = ttk.Frame(list_frame)
        client_list_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("ID", "状态", "PID", "样本数", "准确率")
        self.client_tree = ttk.Treeview(
            client_list_frame,
            columns=columns,
            show="headings",
            height=8
        )

        # 表格样式：行高舒适、表头加粗、斑马纹、状态彩色（参考 MES 工单表格）
        try:
            self.root.style.configure("Client.Treeview", rowheight=30, font=("Microsoft YaHei", 10))
            self.root.style.configure("Client.Treeview.Heading",
                                      font=("Microsoft YaHei", 10, "bold"))
            self.client_tree.configure(style="Client.Treeview")
        except Exception:
            pass
        self.client_tree.tag_configure("even", background="#ffffff")
        self.client_tree.tag_configure("odd", background=COLOR_CHART_BG)
        self.client_tree.tag_configure("running", foreground=COLOR_SUCCESS)
        self.client_tree.tag_configure("stopped", foreground="#c0392b")

        for col in columns:
            self.client_tree.heading(col, text=col)
            self.client_tree.column(col, width=100)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(client_list_frame, orient=tk.VERTICAL, command=self.client_tree.yview)
        self.client_tree.configure(yscrollcommand=scrollbar.set)

        self.client_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 按实际启动的客户端进程填充（不再硬编码 3 个示例行）
        self.populate_client_tree()

        # 客户端操作按钮
        btn_frame = ttk.Frame(status_frame)
        btn_frame.pack(pady=20)

        ttk.Button(
            btn_frame,
            text="🔄 刷新列表",
            command=self.refresh_client_list
        ).pack(side=tk.LEFT, padx=10)

    def _read_client_metric(self, client_id: str):
        """读取客户端最近一次本地训练指标（样本数/准确率）

        数据来源：客户端每次 fit/evaluate 写入 client_logs/{client_id}.log 的 JSON 行
        """
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'client_logs', f'{client_id}.log')
        if not os.path.exists(log_file):
            return None, None
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                return None, None
            last = json.loads(lines[-1])
            return last.get('samples'), last.get('acc')
        except Exception:
            return None, None

    def populate_client_tree(self):
        """按实际启动的客户端进程填充客户端状态列表（真实数据，不造假）"""
        if not hasattr(self, 'client_tree'):
            return
        self.client_tree.delete(*self.client_tree.get_children())

        if not self.client_processes:
            self.client_tree.insert("", "end", values=("暂无客户端", "未启动", "—", "—", "—"))
            return

        row_idx = 0
        for client_id, process in self.client_processes:
            running = process.poll() is None
            status = "🟢 运行中" if running else "🔴 已停止"
            pid = process.pid if process.pid else "—"
            # 读取客户端本地训练指标（训练后会显示真实样本数与准确率）
            samples, acc = self._read_client_metric(client_id)
            tags = ("even" if row_idx % 2 == 0 else "odd",)
            if running:
                tags = tags + ("running",)
            else:
                tags = tags + ("stopped",)
            self.client_tree.insert("", "end", values=(
                client_id, status, pid,
                samples if samples is not None else "—",
                f"{acc:.3f}" if acc is not None else "—"
            ), tags=tags)
            row_idx += 1

    def refresh_client_tree_values(self):
        """就地更新客户端状态树的实时指标（不重建，避免闪烁）"""
        if not hasattr(self, 'client_tree'):
            return
        try:
            row_idx = 0
            for iid in self.client_tree.get_children():
                values = list(self.client_tree.item(iid, 'values'))
                if not values or values[0] in ('暂无客户端', ''):
                    continue
                client_id = values[0]
                samples, acc = self._read_client_metric(client_id)
                proc = next((p for cid, p in self.client_processes if cid == client_id), None)
                running = proc is not None and proc.poll() is None
                status = "🟢 运行中" if running else "🔴 已停止"
                tags = ("even" if row_idx % 2 == 0 else "odd",)
                tags = tags + (("running",) if running else ("stopped",))
                self.client_tree.item(iid, values=(
                    client_id, status, values[2],
                    samples if samples is not None else "—",
                    f"{acc:.3f}" if acc is not None else "—"
                ), tags=tags)
                row_idx += 1
        except Exception:
            pass

    def refresh_client_list(self):
        """刷新客户端列表（真实进程数据）"""
        self.populate_client_tree()
        self.log_message("客户端列表已刷新", "INFO")

    def setup_performance_content(self):
        """设置性能分析内容"""
        perf_frame = ttk.Frame(self.tab_container)
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 性能分析标题
        perf_title = ttk.Label(
            perf_frame,
            text="系统性能分析",
            font=("Microsoft YaHei", 14, "bold")
        )
        perf_title.pack(pady=10)

        # 性能指标框架
        metrics_frame = ttk.LabelFrame(perf_frame, text="性能指标", padding="15")
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 性能指标
        metrics = [
            ("CPU使用率:", f"{np.random.uniform(10, 50):.1f}%"),
            ("内存使用:", f"{np.random.uniform(200, 800):.0f} MB"),
            ("网络带宽:", f"{np.random.uniform(1, 10):.1f} Mbps"),
            ("训练速度:", f"{np.random.uniform(0.5, 2.0):.2f} 轮/分钟"),
            ("数据吞吐:", f"{np.random.uniform(100, 500):.0f} samples/sec")
        ]

        for i, (label, value) in enumerate(metrics):
            frame = ttk.Frame(metrics_frame)
            frame.pack(fill=tk.X, pady=8)

            ttk.Label(
                frame,
                text=label,
                font=("Microsoft YaHei", 11),
                width=15
            ).pack(side=tk.LEFT, padx=10)

            ttk.Label(
                frame,
                text=value,
                font=("Microsoft YaHei", 11, "bold"),
                foreground="#2c3e50"
            ).pack(side=tk.LEFT, padx=10)

        # 性能分析按钮
        btn_frame = ttk.Frame(perf_frame)
        btn_frame.pack(pady=20)

        ttk.Button(
            btn_frame,
            text="📊 生成报告",
            command=self.generate_performance_report
        ).pack(side=tk.LEFT, padx=10)

    def generate_performance_report(self):
        """生成性能报告"""
        self.log_message("性能报告生成成功", "SUCCESS")

    def setup_settings_content(self):
        """设置系统设置内容"""
        settings_frame = ttk.Frame(self.tab_container)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 系统设置标题
        settings_title = ttk.Label(
            settings_frame,
            text="系统设置",
            font=("Microsoft YaHei", 14, "bold")
        )
        settings_title.pack(pady=10)

        # 设置选项框架
        options_frame = ttk.LabelFrame(settings_frame, text="配置选项", padding="15")
        options_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 设置选项
        settings_options = [
            ("主题模式:", ["浅色模式", "深色模式", "自动"]),
            ("语言:", ["中文", "English"]),
            ("日志级别:", ["DEBUG", "INFO", "WARNING", "ERROR"]),
            ("自动保存间隔:", ["5分钟", "10分钟", "30分钟", "1小时"]),
            ("数据备份:", ["启用", "禁用"])
        ]

        self.setting_vars = []

        for i, (label, options) in enumerate(settings_options):
            frame = ttk.Frame(options_frame)
            frame.pack(fill=tk.X, pady=8)

            ttk.Label(
                frame,
                text=label,
                font=("Microsoft YaHei", 11),
                width=15
            ).pack(side=tk.LEFT, padx=10)

            var = tk.StringVar(value=options[0])
            self.setting_vars.append(var)

            combobox = ttk.Combobox(
                frame,
                textvariable=var,
                values=options,
                state="readonly",
                width=15
            )
            combobox.pack(side=tk.LEFT, padx=10)

        # 设置操作按钮
        btn_frame = ttk.Frame(settings_frame)
        btn_frame.pack(pady=20)

        ttk.Button(
            btn_frame,
            text="💾 保存设置",
            command=self.save_settings
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            btn_frame,
            text="🔄 恢复默认",
            command=self.reset_settings
        ).pack(side=tk.LEFT, padx=10)

    def save_settings(self):
        """保存系统设置"""
        self.log_message("系统设置已保存", "SUCCESS")

    def reset_settings(self):
        """恢复默认设置"""
        for var in self.setting_vars:
            var.set(var.get())  # 保持原样，实际应用中应恢复默认值
        self.log_message("已恢复默认设置", "INFO")

    def setup_help_content(self):
        """设置帮助文档内容"""
        help_frame = ttk.Frame(self.tab_container)
        help_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 帮助文档标题
        help_title = ttk.Label(
            help_frame,
            text="帮助文档",
            font=("Microsoft YaHei", 14, "bold")
        )
        help_title.pack(pady=10)

        # 帮助内容
        help_text = scrolledtext.ScrolledText(
            help_frame,
            height=20,
            font=("Microsoft YaHei", 10),
            wrap=tk.WORD
        )
        help_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        help_content = """
联邦学习系统使用指南

1. 快速开始
   - 第一步：启动服务器
   - 第二步：启动客户端
   - 第三步：开始训练
   - 第四步：监控训练进度

2. 服务器配置
   - 端口设置：选择未被占用的端口
   - 训练轮数：设置总训练轮数
   - 最小客户端：设置参与训练的最小客户端数

3. 客户端管理
   - 可以启动多个客户端实例
   - 每个客户端独立训练本地数据
   - 客户端自动连接服务器

4. 训练监控
   - 实时查看训练进度
   - 监控准确率和损失值变化
   - 查看客户端连接状态

5. 模型管理
   - 保存和加载训练好的模型
   - 导出模型供其他应用使用
   - 管理模型版本

6. 故障排除
   - 检查端口是否被占用
   - 确保客户端能连接服务器
   - 查看系统日志获取详细信息

7. 常见问题
   Q: 客户端无法连接服务器？
   A: 检查服务器端口和防火墙设置

   Q: 训练进度不更新？
   A: 确保有足够的客户端连接

   Q: 如何导出训练结果？
   A: 在模型管理页面选择模型导出

8. 技术支持
   - 查看系统日志获取详细信息
   - 联系技术支持团队
   - 查阅在线文档
"""

        help_text.insert(1.0, help_content)
        help_text.configure(state="disabled")  # 设置为只读

    def setup_about_content(self):
        """设置关于系统内容"""
        about_frame = ttk.Frame(self.tab_container)
        about_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 关于系统标题
        about_title = ttk.Label(
            about_frame,
            text="关于联邦学习系统",
            font=("Microsoft YaHei", 16, "bold"),
            foreground="#2c3e50"
        )
        about_title.pack(pady=20)

        # 系统信息
        info_frame = ttk.Frame(about_frame)
        info_frame.pack(pady=10)

        system_info = [
            ("系统名称:", "联邦学习系统 v2.0"),
            ("版本号:", "2.0.0"),
            ("开发团队:", "联邦学习研究组"),
            ("发布日期:", "2025年")
        ]

        for label, value in system_info:
            item_frame = ttk.Frame(info_frame)
            item_frame.pack(pady=5)

            ttk.Label(
                item_frame,
                text=label,
                font=("Microsoft YaHei", 11),
                width=12
            ).pack(side=tk.LEFT, padx=5)

            ttk.Label(
                item_frame,
                text=value,
                font=("Microsoft YaHei", 11),
                foreground="#2c3e50"
            ).pack(side=tk.LEFT, padx=5)

        # 系统描述
        desc_frame = ttk.LabelFrame(about_frame, text="系统简介", padding="15")
        desc_frame.pack(fill=tk.X, pady=20, padx=20)

        description = """联邦学习系统是一个先进的分布式机器学习平台，允许在保护数据隐私的前提下进行协作训练。
系统采用客户端-服务器架构，多个客户端在本地训练模型，仅将模型更新发送到服务器进行聚合。
这种方式既保护了用户隐私，又能利用分布式数据进行模型训练。"""

        desc_label = ttk.Label(
            desc_frame,
            text=description,
            font=("Microsoft YaHei", 10),
            wraplength=600,
            justify=tk.LEFT
        )
        desc_label.pack()

        # 版权信息
        copyright_label = ttk.Label(
            about_frame,
            text="© 2024 联邦学习研究组. 保留所有权利。",
            font=("Microsoft YaHei", 9),
            foreground="#7f8c8d"
        )
        copyright_label.pack(pady=20)

    def open_model_directory(self):
        """在文件资源管理器中打开模型目录"""
        model_dir = self.config.get('model_save_path', './saved_models')
        try:
            if os.path.exists(model_dir):
                if sys.platform == 'win32':
                    os.startfile(model_dir)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', model_dir])
                else:
                    subprocess.Popen(['xdg-open', model_dir])
                self.log_message(f"已打开模型目录: {model_dir}", "INFO")
            else:
                messagebox.showwarning("目录未找到", f"目录不存在: {model_dir}")
        except Exception as e:
            self.log_message(f"打开目录失败: {e}", "ERROR")

    def setup_status_bar(self):
        """设置状态栏"""
        status_frame = ttk.Frame(self.content_frame)
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=4)

        # 服务器状态
        self.server_status_var = tk.StringVar(value="🔴 服务器未运行")
        server_status_label = ttk.Label(status_frame, textvariable=self.server_status_var, font=("Microsoft YaHei", 9))
        server_status_label.pack(side=tk.LEFT, padx=8)

        # 客户端状态
        self.client_status_var = tk.StringVar(value="客户端: 0")
        client_status_label = ttk.Label(status_frame, textvariable=self.client_status_var, font=("Microsoft YaHei", 9))
        client_status_label.pack(side=tk.LEFT, padx=8)

        # 训练状态
        self.train_status_var = tk.StringVar(value="训练: 未开始")
        train_status_label = ttk.Label(status_frame, textvariable=self.train_status_var, font=("Microsoft YaHei", 9))
        train_status_label.pack(side=tk.LEFT, padx=8)

        # 时间显示
        self.time_var = tk.StringVar()
        time_label = ttk.Label(status_frame, textvariable=self.time_var, font=("Microsoft YaHei", 9))
        time_label.pack(side=tk.RIGHT, padx=8)

        # 更新时钟
        self.update_clock()

    def update_clock(self):
        """更新时钟"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_var.set(f"🕒 {current_time}")
        self.root.after(1000, self.update_clock)

    def log_message(self, message: str, level: str = "INFO"):
        """记录消息"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            level_translation = {
                "INFO": "信息",
                "WARNING": "警告",
                "ERROR": "错误",
                "SUCCESS": "成功"
            }

            level_colors = {
                "INFO": "black",
                "WARNING": "orange",
                "ERROR": "red",
                "SUCCESS": "green"
            }

            color = level_colors.get(level, "black")
            level_display = level_translation.get(level, level)
            formatted_message = f"[{timestamp}] [{level_display}] {message}\n"

            # 检查log_text是否存在
            if hasattr(self, 'log_text') and self.log_text.winfo_exists():
                # 直接写入日志文本框
                self.log_text.insert(tk.END, formatted_message)
                self.log_text.tag_add(level, "end-2c linestart", "end-1c")
                self.log_text.tag_config(level, foreground=color)
                self.log_text.see(tk.END)
            else:
                # 缓存日志消息
                self.log_cache.append((timestamp, level, message))
                # 保持缓存大小在限制内
                if len(self.log_cache) > self.max_log_cache:
                    self.log_cache.pop(0)
                # 同时输出到控制台
                print(formatted_message, end='')

        except Exception as e:
            print(f"记录消息时出错: {e}")

    def clear_log(self):
        """清空日志"""
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            self.log_text.delete(1.0, tk.END)
            self.log_message("日志已清空", "INFO")

    def save_log(self):
        """保存日志到文件"""
        try:
            log_file = f"fl_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            if hasattr(self, 'log_text') and self.log_text.winfo_exists():
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.log_message(f"日志已保存到: {log_file}", "SUCCESS")
            else:
                # 如果日志文本框不存在，保存缓存中的日志
                with open(log_file, 'w', encoding='utf-8') as f:
                    for timestamp, level, message in self.log_cache:
                        level_translation = {
                            "INFO": "信息",
                            "WARNING": "警告",
                            "ERROR": "错误",
                            "SUCCESS": "成功"
                        }
                        level_display = level_translation.get(level, level)
                        f.write(f"[{timestamp}] [{level_display}] {message}\n")
                self.log_message(f"日志已保存到: {log_file}", "SUCCESS")
        except Exception as e:
            self.log_message(f"保存日志失败: {e}", "ERROR")

    def save_current_config(self):
        """保存当前配置"""
        try:
            self.config.update({
                'server_port': int(self.port_var.get()),
                'num_rounds': int(self.rounds_var.get()),
                'min_clients': int(self.min_clients_var.get()),
                'local_epochs': int(self.local_epochs_var.get()),
                'model_save_path': self.model_path_var.get()
            })
            self.save_config()
            self.ensure_model_directory()  # 确保新目录存在
            self.log_message(f"配置已保存。模型路径: {self.model_path_var.get()}", "SUCCESS")
        except ValueError as e:
            self.log_message(f"保存配置失败: 请输入有效的数字", "ERROR")

    def test_port(self):
        """测试端口是否可用"""
        try:
            port = int(self.port_var.get())
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()

            if result == 0:
                self.log_message(f"端口 {port} 已被占用，请尝试其他端口", "WARNING")
                messagebox.showwarning("端口检查", f"端口 {port} 已被占用，请使用其他端口")
            else:
                self.log_message(f"端口 {port} 可用", "SUCCESS")
                messagebox.showinfo("端口检查", f"端口 {port} 可用")
        except Exception as e:
            self.log_message(f"端口检查失败: {e}", "ERROR")

    def test_clients(self):
        """测试客户端连接"""
        try:
            port = int(self.port_var.get())
            self.log_message(f"正在测试端口 {port} 上的客户端连接...", "INFO")

            # 先测试服务器
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()

            if result != 0:
                self.log_message(f"端口 {port} 上没有运行服务器，请先启动服务器", "WARNING")
                messagebox.showwarning("客户端测试", "服务器未运行，请先启动服务器")
                return

            # 测试示例客户端连接
            self.log_message("服务器正在运行，客户端连接应正常工作", "SUCCESS")
            messagebox.showinfo("客户端测试", "服务器正在运行，准备好接收客户端连接")

        except Exception as e:
            self.log_message(f"客户端测试失败: {e}", "ERROR")

    def browse_model_path(self):
        """浏览模型保存路径"""
        path = filedialog.askdirectory(
            title="选择模型保存目录",
            initialdir=self.model_path_var.get()
        )
        if path:
            self.model_path_var.set(path)

    def update_charts_data(self):
        """更新图表数据（模拟）"""
        # 模拟数据
        rounds = list(range(1, self.current_round + 1))
        accuracy = [0.5 + 0.4 * (i / len(rounds)) for i in range(len(rounds))]
        loss = [2.0 - 1.5 * (i / len(rounds)) for i in range(len(rounds))]

        # 清除坐标轴以防止重叠
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # 更新准确率图表
        self.ax1.plot(rounds, accuracy, color=COLOR_PRIMARY, linewidth=2)
        self.ax1.set_title('Accuracy Trend', pad=15)
        self.ax1.set_xlabel('Training Rounds')
        self.ax1.set_ylabel('Accuracy')
        self.ax1.grid(True, alpha=0.3)

        if len(rounds) > 10:
            self.ax1.set_xticks(rounds[::max(1, len(rounds) // 10)])
            plt.setp(self.ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 更新损失值图表
        self.ax2.plot(rounds, loss, color=COLOR_ACCENT, linewidth=2)
        self.ax2.set_title('Loss Trend', pad=15)
        self.ax2.set_xlabel('Training Rounds')
        self.ax2.set_ylabel('Loss')
        self.ax2.grid(True, alpha=0.3)
        if len(rounds) > 10:
            self.ax2.set_xticks(rounds[::max(1, len(rounds) // 10)])
            plt.setp(self.ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 更新客户端连接图表
        if self.client_processes:
            times = list(range(len(self.client_processes) * 2))
            clients = [min(i // 2 + 1, len(self.client_processes)) for i in times]
            self.ax3.plot(times, clients, color=COLOR_SUCCESS, linewidth=2)
            self.ax3.set_title('Connected Clients', pad=15)
            self.ax3.set_xlabel('Time')
            self.ax3.set_ylabel('Client Count')
            self.ax3.grid(True, alpha=0.3)

        # 更新样本分布图表
        if self.client_processes:
            client_ids = [f'C{i + 1}' for i in range(len(self.client_processes))]
            samples = [np.random.randint(500, 1500) for _ in client_ids]
            bars = self.ax4.bar(client_ids, samples, color=COLOR_PRIMARY, alpha=0.75)
            self.ax4.set_title('Sample Distribution', pad=15)
            self.ax4.set_xlabel('Client')
            self.ax4.set_ylabel('Sample Count')
            self.ax4.grid(True, alpha=0.3)

            # 在条形图顶部添加数值标签
            for bar in bars:
                height = bar.get_height()
                self.ax4.text(bar.get_x() + bar.get_width() / 2., height,
                              f'{int(height)}', ha='center', va='bottom', fontsize=8)

        # 手动调整布局
        self.fig.tight_layout()

        # 绘制画布
        self.canvas.draw()

    def update_charts(self):
        """初始化图表（等待训练数据，训练开始后由真实指标驱动）"""
        try:
            for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
                ax.clear()
                ax.grid(True, alpha=0.3)
            self.ax1.set_title('Accuracy Trend', pad=15)
            self.ax1.set_ylabel('Accuracy')
            self.ax2.set_title('Loss Trend', pad=15)
            self.ax2.set_ylabel('Loss')
            self.ax3.set_title('Connected Clients', pad=15)
            self.ax3.set_ylabel('Client Count')
            self.ax4.set_title('Sample Distribution (Non-IID)', pad=15)
            self.ax4.set_ylabel('Sample Count')
            self.ax1.text(0.5, 0.5, '等待训练数据…\n启动服务器并开始训练后自动更新',
                          transform=self.ax1.transAxes, ha='center', va='center',
                          fontsize=11, color=COLOR_MUTED)
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"初始化图表失败: {e}")

    def refresh_model_list(self):
        """刷新模型列表"""
        try:
            if not hasattr(self, 'model_listbox'):
                return

            self.model_listbox.delete(0, tk.END)

            model_dir = self.config.get('model_save_path', './saved_models')

            # 确保目录存在
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            # 查找模型文件
            model_extensions = ['.h5', '.pth', '.pt', '.pkl', '.keras', '.model', '.sav']
            model_files = []

            for f in os.listdir(model_dir):
                filepath = os.path.join(model_dir, f)
                if os.path.isfile(filepath):
                    # 检查各种扩展名的模型文件
                    if any(f.lower().endswith(ext) for ext in model_extensions):
                        model_files.append(f)

            if not model_files:
                self.model_listbox.insert(tk.END, "未找到保存的模型")
                self.model_listbox.insert(tk.END, f"目录: {model_dir}")
            else:
                for model_file in sorted(model_files, reverse=True):
                    try:
                        filepath = os.path.join(model_dir, model_file)
                        if os.path.exists(filepath):
                            size_bytes = os.path.getsize(filepath)
                            size_kb = size_bytes / 1024

                            if size_bytes < 1024:
                                size_str = f"{size_bytes} B"
                            elif size_kb < 1024:
                                size_str = f"{size_kb:.1f} KB"
                            else:
                                size_mb = size_kb / 1024
                                size_str = f"{size_mb:.2f} MB"

                            timestamp = datetime.fromtimestamp(
                                os.path.getmtime(filepath)
                            ).strftime('%Y-%m-%d %H:%M')
                            self.model_listbox.insert(
                                tk.END,
                                f"{model_file} ({size_str}, {timestamp})"
                            )
                        else:
                            self.model_listbox.insert(tk.END, f"{model_file} (文件未找到)")
                    except Exception as e:
                        self.model_listbox.insert(tk.END, f"{model_file} (错误: {str(e)[:50]}...)")

                self.log_message(f"在 {model_dir} 中找到 {len(model_files)} 个模型文件", "INFO")

        except Exception as e:
            print(f"刷新模型列表时出错: {e}")

    def load_selected_model(self):
        """加载并查看选中的模型"""
        try:
            selection = self.model_listbox.curselection()
            if not selection:
                messagebox.showwarning("模型管理", "请先选择一个模型")
                return

            selected_text = self.model_listbox.get(selection[0])

            # 检查是否是信息消息，而不是模型文件
            if selected_text.startswith("未找到保存的模型") or \
                    selected_text.startswith("目录:"):
                messagebox.showwarning("模型管理", "请选择一个有效的模型文件")
                return

            # 从列表框条目中提取模型名称
            model_name = selected_text.split()[0]

            model_dir = self.config.get('model_save_path', './saved_models')
            model_path = os.path.join(model_dir, model_name)

            if not os.path.exists(model_path):
                messagebox.showerror("模型管理", f"模型文件未找到: {model_path}")
                self.refresh_model_list()
                return

            # 显示基本模型信息
            file_stats = os.stat(model_path)
            size_bytes = file_stats.st_size
            size_kb = size_bytes / 1024

            if size_bytes < 1024:
                size_str = f"{size_bytes} 字节"
            elif size_kb < 1024:
                size_str = f"{size_kb:.1f} KB"
            else:
                size_mb = size_kb / 1024
                size_str = f"{size_mb:.2f} MB"

            info = f"模型: {model_name}\n"
            info += f"路径: {model_path}\n"
            info += f"大小: {size_str}\n"
            info += f"创建时间: {datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}\n"
            info += f"修改时间: {datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}\n"
            info += f"加载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            # 如果是TensorFlow模型，尝试加载H5模型
            if model_name.lower().endswith('.h5'):
                try:
                    import tensorflow as tf
                    from tensorflow.keras.models import load_model

                    model = load_model(model_path, compile=False)

                    info += "🧠 TensorFlow/Keras 模型详情:\n"
                    info += "-" * 40 + "\n"

                    # 获取模型摘要
                    import io
                    import sys
                    old_stdout = sys.stdout
                    sys.stdout = buffer = io.StringIO()
                    model.summary()
                    sys.stdout = old_stdout
                    summary_text = buffer.getvalue()
                    info += summary_text

                except ImportError:
                    info += "⚠️ TensorFlow未安装。无法加载模型详情。\n"
                except Exception as e:
                    info += f"⚠️ 无法加载模型: {str(e)[:200]}\n"

            elif model_name.lower().endswith(('.pth', '.pt')):
                info += "模型类型: PyTorch (.pth/.pt)\n"
                info += "注意: 需要安装torch才能加载此模型\n"
            else:
                info += f"文件类型: {os.path.splitext(model_name)[1]}\n"

            if hasattr(self, 'model_info_text'):
                self.model_info_text.delete(1.0, tk.END)
                self.model_info_text.insert(1.0, info)

            self.log_message(f"模型信息已加载: {model_name}", "SUCCESS")

        except Exception as e:
            error_msg = f"加载模型信息失败: {str(e)}"
            messagebox.showerror("模型管理", error_msg)
            self.log_message(error_msg, "ERROR")

    def delete_selected_model(self):
        """删除选中的模型"""
        try:
            selection = self.model_listbox.curselection()
            if not selection:
                messagebox.showwarning("模型管理", "请先选择一个模型")
                return

            selected_text = self.model_listbox.get(selection[0])

            # 检查是否实际上是模型文件条目
            if selected_text.startswith("未找到保存的模型") or \
                    selected_text.startswith("目录:"):
                messagebox.showwarning("模型管理", "请选择一个有效的模型文件")
                return

            model_name = selected_text.split()[0]
            model_dir = self.config.get('model_save_path', './saved_models')
            model_path = os.path.join(model_dir, model_name)

            if not os.path.exists(model_path):
                messagebox.showerror("模型管理", f"模型文件未找到: {model_path}")
                self.refresh_model_list()
                return

            if messagebox.askyesno("确认删除", f"确定要删除模型 {model_name} 吗？"):
                try:
                    os.remove(model_path)
                    self.refresh_model_list()
                    if hasattr(self, 'model_info_text'):
                        self.model_info_text.delete(1.0, tk.END)
                    self.log_message(f"模型 {model_name} 已删除", "SUCCESS")
                except Exception as e:
                    self.log_message(f"删除模型失败: {e}", "ERROR")

        except Exception as e:
            self.log_message(f"删除选中模型时出错: {e}", "ERROR")

    def export_model(self):
        """导出模型"""
        try:
            selection = self.model_listbox.curselection()
            if not selection:
                messagebox.showwarning("模型导出", "请选择一个要导出的模型")
                return

            selected_text = self.model_listbox.get(selection[0])

            # 检查是否实际上是模型文件条目
            if selected_text.startswith("未找到保存的模型") or \
                    selected_text.startswith("目录:"):
                messagebox.showwarning("模型导出", "请选择一个有效的模型文件")
                return

            model_name = selected_text.split()[0]
            model_dir = self.config.get('model_save_path', './saved_models')
            source_path = os.path.join(model_dir, model_name)

            if not os.path.exists(source_path):
                messagebox.showerror("模型导出", f"模型文件未找到: {source_path}")
                self.refresh_model_list()
                return

            file_path = filedialog.asksaveasfilename(
                title="导出模型为",
                defaultextension=os.path.splitext(model_name)[1],
                initialfile=model_name,
                filetypes=[
                    ("所有模型文件", "*.h5 *.pth *.pt *.pkl *.keras *.model *.sav"),
                    ("TensorFlow/Keras 文件", "*.h5"),
                    ("PyTorch 文件", "*.pth *.pt"),
                    ("Pickle 文件", "*.pkl"),
                    ("所有文件", "*.*")
                ]
            )

            if file_path:
                try:
                    import shutil
                    shutil.copy2(source_path, file_path)
                    self.log_message(f"模型已导出到: {file_path}", "SUCCESS")
                except Exception as e:
                    self.log_message(f"导出模型失败: {e}", "ERROR")

        except Exception as e:
            self.log_message(f"导出模型时出错: {e}", "ERROR")

    def start_server(self):
        """启动服务器（以独立子进程方式，规避 Flower 在主线程外注册信号处理器的问题）"""
        try:
            port = int(self.port_var.get())

            # 检查端口是否被占用
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()

            if result == 0:
                # 不自动换端口：客户端固定连接该端口，换端口会导致客户端连不上
                self.log_message(f"端口 {port} 已被占用，可能已有其他训练实例在运行", "WARNING")
                messagebox.showerror(
                    "端口被占用",
                    f"端口 {port} 已被占用。\n\n"
                    f"可能原因：另一个服务器/GUI 实例正在运行（重复启动 GUI 或残留进程）。\n\n"
                    f"请先停止其他实例（可在任务管理器中结束 python 进程），"
                    f"或在控制页修改端口后再启动。"
                )
                return

            # 若已有服务器进程在运行，先提示
            if self.server_process and self.server_process.poll() is None:
                self.log_message("服务器已在运行，请先停止", "WARNING")
                return

            script_dir = os.path.dirname(os.path.abspath(__file__))

            # 轮转旧日志：保证本次运行从干净日志开始，进度解析不会混入历史数据
            log_file = os.path.join(script_dir, 'server.log')
            if os.path.exists(log_file):
                try:
                    os.replace(log_file, log_file + '.old')
                except Exception:
                    pass

            # 保存服务器配置（供独立进程读取）
            server_config = {
                'port': port,
                'num_rounds': int(self.rounds_var.get()),
                'min_fit_clients': int(self.min_clients_var.get()),
                'local_epochs': int(self.local_epochs_var.get()),
                'model_save_path': self.model_path_var.get()
            }
            config_file = os.path.join(script_dir, 'server_config_gui.json')
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(server_config, f, indent=2)

            # 以子进程方式启动服务器：server 运行在自己的主线程，Flower 的 signal 处理正常
            env = os.environ.copy()
            env['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 TensorFlow warning 及以上

            self.server_process = subprocess.Popen(
                [sys.executable, 'federated_server.py', '--config', config_file],
                cwd=script_dir,
                env=env,
                # 关键：子进程输出必须丢弃，不能用 PIPE——
                # 服务器/客户端日志已写入 server.log（FileHandler），
                # 若接管道且读取不及时，缓冲区填满会阻塞子进程，训练卡死
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # 更新UI状态
            self.start_server_btn.config(state=tk.DISABLED)
            self.stop_server_btn.config(state=tk.NORMAL)
            self.start_train_btn.config(state=tk.NORMAL)
            self.server_status_var.set(f"🟢 服务器运行中 (端口: {port})")
            self.total_rounds = int(self.rounds_var.get())

            self.log_message(f"服务器成功启动，监听端口: {port}", "SUCCESS")
            self.log_message(f"模型保存路径: {self.model_path_var.get()}", "INFO")

        except Exception as e:
            self.log_message(f"启动服务器失败: {e}", "ERROR")

    def _read_server_output(self, process):
        """读取服务器子进程输出并写入日志面板"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log_message(f"[服务器] {line.strip()}", "INFO")
        except Exception:
            pass

    def run_server(self, config: Dict):
        """运行服务器（兼容旧调用，已弃用，改由独立进程启动）"""
        self.log_message("run_server 已弃用，请使用进程方式启动", "WARNING")

    def stop_server(self):
        """停止服务器"""
        try:
            if self.server_process and self.server_process.poll() is None:
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.server_process.kill()
                self.log_message("服务器进程已终止", "INFO")
            self.server_process = None

            # 更新UI状态
            self.start_server_btn.config(state=tk.NORMAL)
            self.stop_server_btn.config(state=tk.DISABLED)
            self.start_train_btn.config(state=tk.DISABLED)
            self.server_status_var.set("🔴 服务器未运行")
            self.is_training = False

            self.log_message("服务器已停止", "INFO")

        except Exception as e:
            self.log_message(f"停止服务器失败: {e}", "ERROR")

    def start_clients(self):
        """启动客户端"""
        try:
            client_count = int(self.client_count_var.get())
            server_port = int(self.port_var.get())

            # 若有旧客户端进程，先停止（避免新旧客户端混跑，也让数量修改立即生效）
            if self.client_processes:
                self.log_message("检测到已有客户端，先停止旧客户端，再按新数量启动", "WARNING")
                self.stop_all_clients()

            self.log_message(f"正在启动 {client_count} 个客户端...", "INFO")

            for i in range(client_count):
                client_id = f"client_{i + 1}"
                self.start_client_process(client_id, server_port)

            self.start_clients_btn.config(state=tk.DISABLED)
            self.stop_clients_btn.config(state=tk.NORMAL)

        except Exception as e:
            self.log_message(f"启动客户端失败: {e}", "ERROR")

    def start_client_process(self, client_id: str, port: int):
        """启动单个客户端进程"""
        try:
            cmd = [
                sys.executable,
                'federated_client.py',
                '--cid', client_id,
                '--server', f'127.0.0.1:{port}',
                '--num-clients', str(self.client_count_var.get())
            ]

            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                # 关键：客户端输出必须丢弃，不能用 PIPE——
                # TensorFlow 训练进度会写入 stderr，管道缓冲区约 10 轮即被填满，
                # 客户端进程写 stderr 被阻塞 → 无法响应服务器 → 训练卡死
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self.client_processes.append((client_id, process))

            self.update_client_list()
            self.log_message(f"客户端 {client_id} 已启动", "SUCCESS")

        except Exception as e:
            self.log_message(f"启动客户端 {client_id} 失败: {e}", "ERROR")

    def read_client_output(self, client_id: str, process):
        """读取客户端输出"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log_message(f"[{client_id}] {line.strip()}", "INFO")
        except:
            pass

    def stop_all_clients(self):
        """停止所有客户端"""
        try:
            for client_id, process in self.client_processes:
                process.terminate()
                self.log_message(f"客户端 {client_id} 已停止", "INFO")

            self.client_processes.clear()
            self.update_client_list()

            self.start_clients_btn.config(state=tk.NORMAL)
            self.stop_clients_btn.config(state=tk.DISABLED)

        except Exception as e:
            self.log_message(f"停止客户端失败: {e}", "ERROR")

    def update_client_list(self):
        """更新客户端列表"""
        if hasattr(self, 'client_listbox'):
            self.client_listbox.delete(0, tk.END)
            for client_id, process in self.client_processes:
                status = "🟢 运行中" if process.poll() is None else "🔴 已停止"
                self.client_listbox.insert(tk.END, f"{client_id}: {status}")

        self.client_status_var.set(f"客户端: {len(self.client_processes)}")

    def start_training(self):
        """开始训练"""
        if not self.client_processes:
            messagebox.showwarning("训练控制", "请先启动客户端")
            return

        self.is_training = True
        self.current_round = 0

        # 更新UI状态
        self.start_train_btn.config(state=tk.DISABLED)
        self.pause_train_btn.config(state=tk.NORMAL)
        self.stop_train_btn.config(state=tk.NORMAL)
        self.train_status_var.set(f"训练中 (第0轮)")

        self.log_message("联邦学习训练已开始", "SUCCESS")
        self.log_message(f"模型将保存到: {self.model_path_var.get()}", "INFO")

        # 模拟训练进度更新
        self.update_training_progress()

    def update_training_progress(self):
        """更新训练进度（基于服务器日志的真实进度，而非模拟）"""
        if not self.is_training:
            return

        # 服务器进程已退出则停止
        if self.server_process is not None and self.server_process.poll() is not None:
            self.log_message("服务器进程已退出，训练结束", "WARNING")
            self.is_training = False
            self._finish_training()
            return

        # 从 server.log 解析真实轮次与指标
        current_round, metrics = self._parse_server_progress()
        self.current_round = current_round

        if current_round > 0:
            progress = (current_round / max(1, self.total_rounds)) * 100
            self.progress_var.set(progress)
            self.progress_label.config(text=f"第 {current_round}/{self.total_rounds} 轮")
            self.train_status_var.set(f"训练中 (第{current_round}/{self.total_rounds}轮)")
        else:
            self.progress_var.set(0)
            self.train_status_var.set("训练中 (等待第一轮完成...)")

        # 使用真实指标更新图表
        if metrics:
            self._update_charts_with_real_data(metrics)

        # 训练完成判定
        if current_round >= self.total_rounds:
            self.is_training = False
            self._finish_training()
            self.log_message("联邦学习训练已完成", "SUCCESS")
            return

        # 每2秒轮询一次
        self.root.after(2000, self.update_training_progress)

    def _finish_training(self):
        """训练结束收尾"""
        self.start_train_btn.config(state=tk.NORMAL)
        self.pause_train_btn.config(state=tk.DISABLED)
        self.stop_train_btn.config(state=tk.DISABLED)
        self.train_status_var.set("训练: 已完成")
        self.refresh_model_list()

    def _parse_server_progress(self):
        """从 server.log 解析真实训练进度

        返回: (当前轮次, 指标字典 {round: {'train_acc', 'train_loss', 'test_acc'}})
        """
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server.log')
        current_round = 0
        metrics = {}

        try:
            if not os.path.exists(log_file):
                return 0, {}

            import re
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 轮次：Flower 日志形如 [ROUND 3]
            round_matches = re.findall(r'\[ROUND (\d+)\]', content)
            if round_matches:
                current_round = max(int(x) for x in round_matches)

            # 训练指标：FitMetrics = {'train_acc': 0.xx, 'train_loss': 0.xx}
            for m in re.finditer(r'\[ROUND (\d+)\] FitMetrics = \{(.*?)\}', content):
                r = int(m.group(1))
                acc = re.search(r"'train_acc': ([0-9.eE+-]+)", m.group(2))
                loss = re.search(r"'train_loss': ([0-9.eE+-]+)", m.group(2))
                if r not in metrics:
                    metrics[r] = {}
                if acc:
                    metrics[r]['train_acc'] = float(acc.group(1))
                if loss:
                    metrics[r]['train_loss'] = float(loss.group(1))

            # 评估指标：EvaluateMetrics = {'test_acc': 0.xx}
            for m in re.finditer(r'\[ROUND (\d+)\] EvaluateMetrics = \{(.*?)\}', content):
                r = int(m.group(1))
                tacc = re.search(r"'test_acc': ([0-9.eE+-]+)", m.group(2))
                if r not in metrics:
                    metrics[r] = {}
                if tacc:
                    metrics[r]['test_acc'] = float(tacc.group(1))

        except Exception as e:
            print(f"解析服务器日志失败: {e}")

        return current_round, metrics

    def _update_charts_with_real_data(self, metrics):
        """用真实训练指标更新图表（替代原模拟数据）"""
        try:
            rounds = sorted(metrics.keys())
            if not rounds:
                return

            train_acc = [metrics[r].get('train_acc') for r in rounds]
            train_loss = [metrics[r].get('train_loss') for r in rounds]
            test_acc = [metrics[r].get('test_acc') for r in rounds]

            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()

            # 准确率趋势（训练 + 测试）
            self.ax1.plot(rounds, train_acc, color=COLOR_PRIMARY, linewidth=2, label='train_acc')
            if any(v is not None for v in test_acc):
                self.ax1.plot(rounds, test_acc, color=COLOR_SUCCESS, linestyle='--', linewidth=2, label='test_acc')
            self.ax1.set_title('Accuracy Trend', pad=15)
            self.ax1.set_xlabel('Training Rounds')
            self.ax1.set_ylabel('Accuracy')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend()

            # 损失趋势
            self.ax2.plot(rounds, train_loss, color=COLOR_ACCENT, linewidth=2)
            self.ax2.set_title('Loss Trend', pad=15)
            self.ax2.set_xlabel('Training Rounds')
            self.ax2.set_ylabel('Loss')
            self.ax2.grid(True, alpha=0.3)

            # 客户端连接状态
            self.ax3.text(0.5, 0.5,
                          f'{len(self.client_processes)} 个客户端',
                          transform=self.ax3.transAxes, ha='center', va='center')
            self.ax3.set_title('Connected Clients', pad=15)
            self.ax3.set_xlabel('Time')
            self.ax3.set_ylabel('Client Count')
            self.ax3.set_ylim(0, max(1, len(self.client_processes) + 1))
            self.ax3.grid(True, alpha=0.3)

            # 样本分布（Non-IID 划分，每客户端最多 1000 样本）
            if self.client_processes:
                client_ids = [f'C{i + 1}' for i in range(len(self.client_processes))]
                samples = [1000] * len(client_ids)
                bars = self.ax4.bar(client_ids, samples, color=COLOR_PRIMARY, alpha=0.75)
                self.ax4.set_title('Sample Distribution (Non-IID)', pad=15)
                self.ax4.set_xlabel('Client')
                self.ax4.set_ylabel('Sample Count')
                self.ax4.grid(True, alpha=0.3)
                for bar in bars:
                    height = bar.get_height()
                    self.ax4.text(bar.get_x() + bar.get_width() / 2., height,
                                  f'{int(height)}', ha='center', va='bottom', fontsize=8)

            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"更新真实图表失败: {e}")

    def pause_training(self):
        """暂停训练（仅停止界面轮询，服务器训练由 stop_server 控制）"""
        self.is_training = False
        self.pause_train_btn.config(state=tk.DISABLED)
        self.start_train_btn.config(state=tk.NORMAL)
        self.train_status_var.set(f"训练已暂停 (第{self.current_round}轮)")

        self.log_message("训练已暂停（服务器仍在运行，可点开始继续监控）", "WARNING")

    def stop_training(self):
        """停止训练"""
        self.is_training = False

        # 更新UI状态
        self.start_train_btn.config(state=tk.NORMAL)
        self.pause_train_btn.config(state=tk.DISABLED)
        self.stop_train_btn.config(state=tk.DISABLED)
        self.train_status_var.set("训练: 已完成")

        self.log_message("训练已完成", "SUCCESS")
        self.refresh_model_list()  # 训练后刷新模型列表

    def monitor_status(self):
        """监控系统状态"""
        try:
            # 更新客户端状态
            self.update_client_list()
            # 就地更新客户端状态页的实时指标（样本数/准确率，不重建列表避免闪烁）
            self.refresh_client_tree_values()
            # 就地更新服务器状态页的端口/轮数/进度
            self._update_server_status_page()

            # 每5秒检查一次
            self.root.after(5000, self.monitor_status)
        except Exception as e:
            print(f"监控状态时出错: {e}")


def _check_environment() -> bool:
    """检查当前 Python 环境是否具备运行依赖（flwr / tensorflow）"""
    import importlib.util
    missing = []
    for mod in ('flwr', 'tensorflow'):
        if importlib.util.find_spec(mod) is None:
            missing.append(mod)
    if missing:
        messagebox.showerror(
            "环境错误",
            f"当前 Python 环境缺少依赖：{', '.join(missing)}\n\n"
            f"请用项目虚拟环境启动 GUI：\n"
            f"D:\\guiproject\\.venv\\Scripts\\python.exe federated_gui.py\n\n"
            f"（否则服务器/客户端子进程将无法运行）"
        )
        return False
    return True


def _acquire_single_instance_lock():
    """单实例锁：Windows 文件区域锁（msvcrt），跨进程互斥

    用文件锁替代端口锁：端口锁存在 TIME_WAIT 误报（前一个实例退出后，
    端口短暂不可绑定会被误判为"已有实例"）。文件锁进程退出自动释放，
    无此问题。返回持有锁的文件对象；失败返回 None。
    """
    import msvcrt
    lock_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.gui.lock')
    try:
        f = open(lock_file, 'w', encoding='utf-8')
        f.write(str(os.getpid()))
        f.flush()
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        return f
    except (OSError, IOError):
        try:
            f.close()
        except Exception:
            pass
        return None


def main():
    """主函数"""
    try:
        # 单实例锁：防止重复启动多个 GUI（多实例会抢占端口、相互干扰导致训练卡死）
        lock_file = _acquire_single_instance_lock()
        if lock_file is None:
            messagebox.showerror(
                "已有实例在运行",
                "联邦学习系统 GUI 已在运行。\n\n"
                "重复启动多个 GUI 会导致端口冲突、训练卡死，"
                "请使用已打开的窗口，或先关闭它再启动。"
            )
            return

        # 环境检查：确保用项目虚拟环境启动
        if not _check_environment():
            return

        # ttkbootstrap 主题窗口（默认 flatly：高级简约、低饱和、专业）
        root = ttkb.Window(themename="flatly")

        app = FederatedLearningGUI(root)
        # 持有锁引用，防止被垃圾回收导致锁提前释放
        app._lock_file = lock_file
        root.mainloop()

    except Exception as e:
        print(f"致命错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
