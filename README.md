# 联邦学习系统（Federated Learning System）

基于 **Flower + TensorFlow** 的联邦学习课程设计项目：多客户端在不共享原始数据的前提下，通过 **FedAvg** 算法协同训练 Fashion-MNIST 图像分类 CNN 模型，并配套 **PySide6 专业版图形界面**（10 页导航 / 折叠侧边栏 / 深浅双主题 / 真实训练指标监控）。

> 学校：集美大学 · 网络安全专业 · 本科课程设计

```
多客户端（Non-IID 数据） ⇄ gRPC ⇄ 联邦服务器（FedAvg 聚合 + 全局模型）
        ↕                                 ↕
   本地 CNN 训练                     GUI 监控 / 模型管理
```

## 功能特性

- **Non-IID 数据划分**：每个客户端按 `client_id` 分配专属类别样本（每客户端 2 类），模拟真实"数据孤岛"场景
- **FedAvg 聚合**：服务器端实现标准联邦平均策略，支持分数采样（fraction_fit / fraction_evaluate）
- **统一 CNN 架构**：服务器与客户端使用一致模型（Conv2D×2 + Dense×2）
- **PySide6 专业版 GUI**（`federated_gui_pyside.py`）：
  - 10 个功能页：控制面板 / 外观设置 / 模型管理 / 客户端状态 / 性能分析 / 训练监控 / 服务器状态 / 日志查看 / 帮助文档 / 关于系统
  - QScrollArea 侧边栏 + QPropertyAnimation 折叠动画（收起仅图标）
  - QSS 现代浅色科技配色（Hover / Active 指示条 / 状态栏区分），浅色 / 深色主题一键切换
  - 响应式布局：卡片撑满可用区、表单标签右对齐垂直参考线、窗口任意缩放重心稳定
- **真实训练指标**：GUI 训练进度与曲线全部来自 `server.log` 中服务器输出的真实轮次指标，无模拟数据
- **性能分析**：实时 KPI（训练轮次 / 训练准确率 / 测试准确率 / 训练损失 / 客户端数 / 客户端样本），每 3 秒刷新
- **模型管理**：训练过程按 `save_interval` 自动保存全局模型（`.h5`），支持加载 / 删除 / 导出
- **一键启动**：`run_all.py` 自动拉起服务器 + N 个客户端 + GUI；`启动联邦学习GUI.bat` 双击无黑窗启动 PySide6 版

## 环境要求

- Windows 10/11（项目在 Windows 验证）
- Python 3.11（3.10/3.12 亦可，TensorFlow 2.16 官方支持 3.9–3.12）

## 快速开始

### 1. 创建虚拟环境并安装依赖

```powershell
cd federated-learning-system
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

核心依赖已锁定：`flwr==1.9.0`、`tensorflow==2.16.2`、`numpy<2.0`、`pyarrow==15.0.2`、`PySide6`。

### 2. 准备数据集（国内网络必需）

TensorFlow 默认从 Google 下载 Fashion-MNIST，国内常被阻断。项目提供国内镜像下载脚本：

```powershell
.\\.venv\\Scripts\\python scripts\\fetch_fashion_mnist.py
```

脚本从 hf-mirror.com 下载官方 parquet 数据，自动转换为 Keras 缓存所需的 idx gz 文件（约 30 MB，放 `~/.keras/datasets/fashion-mnist/`）。已准备过则跳过。

### 3. 启动方式

**方式一：一键启动（推荐）**

```powershell
python run_all.py
# 按提示输入 1（自动启动 服务器 + 2 个客户端 + GUI）
```

**方式二：GUI 手动控制（推荐演示）**

```powershell
# PySide6 专业版（主推）
python federated_gui_pyside.py
```

GUI 中按顺序：
1. 控制面板 → 设置端口 / 轮数 / 客户端数 → 点 **启动服务器**
2. 点 **启动客户端**（启动 N 个客户端进程）
3. 点 **开始训练** → 切到训练监控查看真实训练曲线，性能分析查看实时 KPI

> 说明：GUI 通过独立子进程启动服务器，规避 Flower 在非主线程注册信号处理器导致的 `signal only works in main thread` 崩溃；所有子进程 `stdout/stderr=DEVNULL` + `CREATE_NO_WINDOW`，无黑窗弹出、无管道阻塞。

**方式三：命令行分离启动（调试用）**

```powershell
# 终端1：服务器
python federated_server.py --port 8080 --num-rounds 20 --min-clients 2

# 终端2/3：客户端（各开一个）
python federated_client.py --cid client_1 --server 127.0.0.1:8080 --num-clients 2
python federated_client.py --cid client_2 --server 127.0.0.1:8080 --num-clients 2
```

## 配置说明

| 文件 | 用途 |
| --- | --- |
| `config.json` | 默认完整配置模板（服务器 / 客户端 / 模型） |
| `gui_config.json` | GUI 界面配置（端口 / 轮数 / 路径，运行时生成） |
| `server_config_gui.json` | GUI 启动服务器时写入的运行时配置（运行时生成） |

服务器参数优先级：**命令行参数 > 配置文件（`--config`）> 内置默认值**。

## 目录结构

```
federated-learning-system/
├── federated_gui_pyside.py   # PySide6 专业版 GUI（主推，10 页导航）
├── federated_gui.py          # Tkinter 版 GUI（ttkbootstrap 主题）
├── federated_server.py       # Flower 服务器：FedAvg + 全局模型 + 指标日志
├── federated_client.py       # Flower 客户端：Non-IID 划分 + 本地 CNN 训练
├── run_all.py                # 一键启动器（进程管理）
├── main.py                   # 依赖安装脚本（阿里云镜像）
├── fix_chinese_font.py       # matplotlib 中文乱码修复
├── requirements.txt          # 依赖清单（锁定版本）
├── config.json               # 配置模板
├── scripts/                  # 验证与工具脚本（见下）
├── tests/                    # 单元测试（pytest）
└── 启动联邦学习GUI.bat        # 双击无黑窗启动
```

## 测试与验证

### 单元测试

```powershell
.\\.venv\\Scripts\\python -m pytest tests/ -v
```

覆盖：Non-IID 数据划分正确性、客户端 ID 解析、标签 / 样本一致性校验、端口占用检测、可用端口分配。

### 验证脚本体系（`scripts/`）

| 脚本 | 验证内容 |
| --- | --- |
| `smoke_test.py` | 一键冒烟：自动拉起服务器 + 2 客户端，跑 N 轮并校验指标与模型保存 |
| `verify_pyside_pages.py` | 10 页导航映射（off-screen 构建） |
| `verify_layout_spec.py` | 布局断言（卡片撑满 / 标签右对齐 / 最小尺寸） |
| `verify_double_click.py` | 二次训练防误判（旧日志残留不得误判"已完成 100%"） |
| `verify_training_flow.py` | 训练流程（启动服务器 / 客户端后进度不得自动变化，点开始训练才动） |
| `verify_charts.py` | 图表逻辑（真实数据重绘 / 重置后无残留曲线） |
| `verify_session_boundary.py` | 会话边界（二次训练 ROUND 重新从 1 开始，丢弃上一会话指标） |
| `verify_full_e2e.py` | 综合端到端：真实 server + 2 客户端完整训练 + 模型落盘 + 二次会话边界（11 项断言） |
| `check_log.py` | server.log 结构与指标解析检查 |

运行全部验证：

```powershell
.\\.venv\\Scripts\\python scripts\\verify_full_e2e.py
```

> 注意：验证脚本会临时改写端口配置并清理 `server.log`，请在无真实训练时运行。

## 已验证结果（2026-10-30）

| 验证项 | 结果 |
| --- | --- |
| 单元测试 | 8/8 通过 |
| 冒烟测试（5 轮 × 2 客户端） | 通过，约 16 秒 |
| 综合 E2E（真实训练 + 二次会话） | 11/11 断言通过 |
| 训练收敛 | train_acc 87.5% → 93.4%（第 3 轮峰值） |
| 泛化趋势 | test_acc 17.5% → 19.6%（逐轮上升） |
| 全局模型保存 | 每 `save_interval` 轮自动保存 `.h5` 成功 |

> test_acc 偏低属预期：Non-IID 强划分（每客户端仅 2 类）+ 低轮数 + CPU 单 epoch。提高轮数（如 20 轮）或放宽类别划分（每客户端 3–4 类）可显著提升。

## 开发历程与硬核 Bug 修复（Git 提交史精选）

1. **训练 10–12 轮卡死（根因：子进程管道灌满）**：GUI 子进程 `stdout/stderr=PIPE` 无消费者，64KB 管道缓冲区灌满后客户端阻塞。修复：所有 `Popen` 一律 `stdout=stderr=DEVNULL` + `CREATE_NO_WINDOW`。
2. **17 轮卡死（根因：跨段贪婪正则回溯）**：`_parse_progress` 由跨段贪婪正则改为**流式逐行解析**（O(n)），全程 GUI 响应正常。
3. **点两次"开始训练"才真正开始（根因：旧日志残留误判）**：会话检测由"当前轮 >= 1"改为"曾出现 >1 轮次后又见 ROUND 1"判定（同轮 Fit/Evaluate 双行不再误判回退）；`_parse_progress` 返回 `new_session` 标志；`start_training` 记录点击基准轮次，旧日志残留且新训练未开始时保持等待，训练一旦推进即正常更新。
4. **二次训练曲线混入旧轮次**：`_parse_progress` 加会话边界——ROUND 重新从 1 开始即丢弃上一会话指标；点"开始训练"自动清空曲线重新记录，训练完成后曲线保留供查看。
5. **未训练时图表残留上一轮曲线**：空指标显示"等待开始训练"占位，绝不残留。

## 已知限制

- 模型以 `.h5`（HDF5）格式保存，TensorFlow 的格式弃用警告不影响功能，后续可迁移原生 `.keras` 格式
- 数据集首次运行需联网下载（约 30 MB），之后缓存在本机
- 当前为 CPU 训练，大规模轮次耗时较长，可调小 `num_rounds` 验证
- Non-IID 划分下 test_acc 起步较低（约 17%），需更多轮次收敛

## License

本项目仅用于学习交流与课程设计展示。
