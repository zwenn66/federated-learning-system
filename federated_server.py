#!/usr/bin/env python3
"""
修复版本联邦学习服务器 - 修复Flower版本兼容性问题
"""

import logging
import socket
import sys
import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import flwr as fl
from flwr.common import Parameters, Scalar, NDArrays
from flwr.server import Server, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager
import tensorflow as tf
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FedServer")


class RobustFedServer:
    """健壮的联邦学习服务器"""

    def __init__(self, config: Dict):
        self.config = config
        self.global_model = None
        self.current_round = 0
        self.client_metrics = []
        self.global_metrics = []
        self.model_save_path = config.get('model_save_path', './saved_models')
        os.makedirs(self.model_save_path, exist_ok=True)

        # 创建全局模型
        self.create_global_model()

    def create_global_model(self):
        """创建全局模型架构"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.global_model = model
        logger.info("全局模型创建完成")

    def save_global_model(self, round_num: int):
        """保存全局模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"global_model_round{round_num}_{timestamp}.h5"
        filepath = os.path.join(self.model_save_path, filename)

        try:
            self.global_model.save(filepath)
            logger.info(f"全局模型已保存到: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return None

    def get_initial_parameters(self) -> Parameters:
        """获取初始参数"""
        if self.global_model is None:
            self.create_global_model()

        weights = self.global_model.get_weights()
        parameters = fl.common.ndarrays_to_parameters(weights)
        return parameters

    def fit_config(self, server_round: int) -> Dict[str, Scalar]:
        """训练配置"""
        return {
            "batch_size": self.config.get('batch_size', 32),
            "local_epochs": self.config.get('local_epochs', 1),
            "learning_rate": self.config.get('learning_rate', 0.01),
            "server_round": server_round
        }

    def evaluate_config(self, server_round: int) -> Dict[str, Scalar]:
        """评估配置"""
        return {"batch_size": self.config.get('batch_size', 32)}

    def weighted_average(self, metrics: List[Tuple[int, Dict]]) -> Dict[str, Scalar]:
        """加权平均指标"""
        if not metrics:
            return {}

        total_examples = sum([num_examples for num_examples, _ in metrics])
        weighted_metrics = {}

        # 检查是否所有metrics都有相同的key
        if metrics and metrics[0][1]:
            for key in metrics[0][1].keys():
                try:
                    weighted_value = sum([
                        num_examples * m[key] for num_examples, m in metrics
                    ])
                    weighted_metrics[key] = weighted_value / total_examples
                except:
                    continue

        return weighted_metrics


def trimmed_mean_aggregate(ndarrays_list, trim_fraction: float = 0.2):
    """坐标级 Trimmed Mean 鲁棒聚合

    对每个权重张量的每个坐标：去掉最大 / 最小各 k 个客户端的值后求平均，
    从而剔除被放大的恶意更新（缩放攻击）对聚合结果的带偏。
    客户端数 < 3 时退化为普通平均（鲁棒聚合需要足够的样本量）。

    - ndarrays_list: List[List[np.ndarray]]，每个元素是一个客户端的完整权重列表
    - trim_fraction: 每端裁剪比例，k = round(n * trim_fraction)
    """
    n = len(ndarrays_list)
    n_layers = len(ndarrays_list[0])
    if n < 3:
        return [np.mean([np.asarray(w[i], dtype=np.float32) for w in ndarrays_list], axis=0)
                for i in range(n_layers)]

    k = max(1, int(round(n * trim_fraction))) if trim_fraction > 0 else 0
    # 保证裁剪后至少保留 2 个客户端
    if 2 * k >= n:
        k = n // 2 - 1 if n % 2 == 0 else n // 2
        k = max(1, k)

    agg = []
    for i in range(n_layers):
        layer_weights = [np.asarray(w[i], dtype=np.float32) for w in ndarrays_list]
        if layer_weights[0].ndim == 0:
            arr = np.array(layer_weights)
            arr.sort()
            agg.append(arr[k:-k].mean() if 2 * k < len(arr) else arr.mean())
        else:
            flat = np.stack([lw.reshape(-1) for lw in layer_weights])  # (n, dim)
            sorted_f = np.sort(flat, axis=0)
            trimmed = sorted_f[k:-k] if 2 * k < n else sorted_f
            agg.append(trimmed.mean(axis=0).reshape(layer_weights[0].shape))
    return agg


def krum_aggregate(ndarrays_list, num_malicious: int = 1):
    """Krum 鲁棒聚合：选择与其余更新最"一致"的单个客户端更新

    原版（Blanchard et al. 2017）：对每个更新，取它与最近的 n-f-2 个其他更新的
    欧氏距离平方和作为得分，得分最小的更新即被选中（多数诚实假设下，离群的
    恶意更新得分必然偏高而被排除）。本实现直接返回被选中客户端的完整权重。

    注：单点 Krum 仅输出一个客户端的更新，不损失后续聚合的简单性；
    当恶意客户端占比过高（≥50%）时"多数诚实"假设失效，防御同样失效。
    """
    n = len(ndarrays_list)
    if n < 3:
        return [np.mean([np.asarray(w[i], dtype=np.float32) for w in ndarrays_list], axis=0)
                for i in range(len(ndarrays_list[0]))]
    f = max(0, min(int(num_malicious), n - 2))

    flat = [np.concatenate([np.asarray(w, dtype=np.float32).reshape(-1)
                            for w in cw]) for cw in ndarrays_list]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(flat[i] - flat[j]))
            dist[i, j] = dist[j, i] = d

    keep = max(1, n - f - 2)  # 与最近的 n-f-2 个更新比较
    scores = []
    for i in range(n):
        others = sorted(dist[i, j] ** 2 for j in range(n) if j != i)
        scores.append(sum(others[:keep]))
    best = int(np.argmin(scores))
    return ndarrays_list[best]


class LoggingFedAvg(FedAvg):
    """带指标日志 + 定时保存 + 可选鲁棒聚合的 FedAvg 策略

    - 每轮训练/评估后将聚合指标写入 server.log（供 GUI 解析真实训练曲线）
    - 每 save_interval 轮保存一次全局模型到 saved_models/
    - defense='trimmed_mean' / 'krum' 时，聚合前先对客户端更新做鲁棒聚合，
      抵御缩放 / 投毒类恶意客户端；defense='none' 保持标准 FedAvg 行为
    - 聚合后把真实全局权重写回 server_instance.global_model，
      确保保存的 .h5 是聚合后的全局模型（而非初始随机模型）
    """

    def __init__(self, server_instance: Optional[RobustFedServer] = None,
                 save_interval: int = 5, defense: str = 'none',
                 trim_fraction: float = 0.2, num_malicious: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._server_instance = server_instance
        self._save_interval = max(1, save_interval)
        self._defense = defense
        self._trim_fraction = trim_fraction
        self._num_malicious = num_malicious

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None
        if self._defense in ('trimmed_mean', 'krum'):
            # ---- 鲁棒聚合路径：先防御，再聚合指标 ----
            ndarrays_list = [fl.common.parameters_to_ndarrays(fit_res.parameters)
                             for _, fit_res in results]
            if self._defense == 'trimmed_mean':
                agg_ndarrays = trimmed_mean_aggregate(ndarrays_list, self._trim_fraction)
            else:
                agg_ndarrays = krum_aggregate(ndarrays_list, self._num_malicious)
            agg_params = fl.common.ndarrays_to_parameters(agg_ndarrays)

            weights_results = [(fit_res.num_examples, fit_res.metrics)
                               for _, fit_res in results]
            metrics_agg = (self.fit_metrics_aggregation_fn(weights_results)
                           if self.fit_metrics_aggregation_fn else None)

            if self._server_instance:
                try:
                    self._server_instance.global_model.set_weights(agg_ndarrays)
                except Exception as e:
                    logger.warning(f"全局模型权重回写失败: {e}")
            if metrics_agg:
                logger.info(f"[ROUND {server_round}] FitMetrics = {metrics_agg}")
            if self._server_instance and server_round % self._save_interval == 0:
                self._server_instance.save_global_model(server_round)
            return agg_params, metrics_agg

        # ---- 标准 FedAvg 路径 ----
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            params, metrics = aggregated
            if metrics:
                logger.info(f"[ROUND {server_round}] FitMetrics = {metrics}")
            # 回写真实全局权重，保证保存的模型是聚合结果
            if self._server_instance:
                try:
                    self._server_instance.global_model.set_weights(
                        fl.common.parameters_to_ndarrays(params))
                except Exception as e:
                    logger.warning(f"全局模型权重回写失败: {e}")
            # 按配置间隔保存全局模型
            if self._server_instance and server_round % self._save_interval == 0:
                self._server_instance.save_global_model(server_round)
        return aggregated

    def aggregate_evaluate(self, server_round: int, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is not None:
            loss, metrics = aggregated
            logger.info(f"[ROUND {server_round}] EvaluateMetrics = {metrics}")
        return aggregated


# 使用正确的FedAvg初始化参数
def create_strategy(server_instance: RobustFedServer, config: Dict) -> FedAvg:
    """创建FedAvg策略，使用兼容的初始化方式，并输出每轮指标日志"""

    # 获取初始参数
    initial_parameters = server_instance.get_initial_parameters()

    # 创建策略 - 使用Flower 1.4+ 的API，输出每轮指标日志并按间隔保存模型
    strategy = LoggingFedAvg(
        server_instance=server_instance,
        save_interval=config.get('save_interval', 5),
        defense=config.get('defense', 'none'),
        trim_fraction=config.get('trim_fraction', 0.2),
        num_malicious=config.get('num_malicious', 1),
        fraction_fit=config.get('fraction_fit', 0.5),
        fraction_evaluate=config.get('fraction_evaluate', 0.5),
        min_fit_clients=config.get('min_fit_clients', 2),
        min_evaluate_clients=config.get('min_evaluate_clients', 2),
        min_available_clients=config.get('min_available_clients', 2),
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=server_instance.weighted_average,
        fit_metrics_aggregation_fn=server_instance.weighted_average,
        on_fit_config_fn=server_instance.fit_config,
        on_evaluate_config_fn=server_instance.evaluate_config,
    )

    return strategy


def check_port_available(port: int, host: str = '127.0.0.1') -> bool:
    """检查端口是否可用（connect 探测，Windows 兼容）

    - 能连上 → 端口被监听 → 不可用（返回 False）
    - 连接被拒/超时 → 无监听 → 可用（返回 True）
    不用 bind 探测：Windows 上 SO_REUSEADDR 或通配/具体地址绑定语义
    会导致对已占用端口误判为可用。
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0
    except OSError:
        return True


def find_available_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """寻找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        if check_port_available(port):
            return port
    raise RuntimeError(f"在端口 {start_port}-{start_port + max_attempts} 范围内找不到可用端口")


def start_server(config: Dict) -> Tuple[bool, str]:
    """启动联邦学习服务器"""
    try:
        # 检查并获取可用端口
        port = config.get('port', 8080)
        if not check_port_available(port):
            port = find_available_port(port)
            config['port'] = port
            logger.info(f"端口 {port - 1} 被占用，使用端口 {port}")

        # 创建服务器实例
        server_instance = RobustFedServer(config)

        # 创建策略 - 修复版本兼容性问题
        strategy = create_strategy(server_instance, config)

        # 创建服务器配置
        server_config = ServerConfig(
            num_rounds=config.get('num_rounds', 10),
            round_timeout=config.get('round_timeout', 600)  # 10分钟超时
        )

        # 启动服务器
        logger.info(f"启动联邦学习服务器，监听端口: {port}")
        logger.info(f"最小客户端数: {config.get('min_fit_clients', 2)}")
        logger.info(f"训练轮数: {config.get('num_rounds', 10)}")

        # 保存配置
        config_file = os.path.join(server_instance.model_save_path, 'server_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 启动Flower服务器
        fl.server.start_server(
            server_address=f"0.0.0.0:{port}",
            config=server_config,
            strategy=strategy,
            grpc_max_message_length=1024 * 1024 * 1024,  # 1GB
            certificates=None  # 不使用TLS
        )

        return True, f"服务器在端口 {port} 启动成功"

    except Exception as e:
        logger.error(f"启动服务器失败: {e}")
        import traceback
        traceback.print_exc()
        return False, f"启动失败: {str(e)}"


def main():
    """主函数（支持命令行参数与配置文件）"""
    import argparse

    parser = argparse.ArgumentParser(description='Fashion-MNIST联邦学习服务器')
    parser.add_argument('--config', default=None, help='配置文件路径（JSON），如 server_config_gui.json')
    parser.add_argument('--port', type=int, default=None, help='监听端口（覆盖配置）')
    parser.add_argument('--num-rounds', type=int, default=None, help='训练轮数（覆盖配置）')
    parser.add_argument('--min-clients', type=int, default=None, help='最小客户端数（覆盖配置）')
    parser.add_argument('--local-epochs', type=int, default=None, help='本地训练轮数（覆盖配置）')
    parser.add_argument('--model-save-path', default=None, help='模型保存路径（覆盖配置）')
    parser.add_argument('--defense', default=None, choices=['none', 'trimmed_mean', 'krum'],
                        help='鲁棒聚合防御：none=标准FedAvg / trimmed_mean=坐标裁剪平均 / krum=单点选择（覆盖配置）')
    parser.add_argument('--trim-fraction', type=float, default=None, help='Trimmed Mean 每端裁剪比例（默认0.2）')
    parser.add_argument('--num-malicious', type=int, default=None, help='Krum 假设的恶意客户端数（默认1）')
    args = parser.parse_args()

    # 服务器配置（默认值）
    config = {
        'port': 8080,
        'num_rounds': 20,
        'fraction_fit': 0.5,
        'fraction_evaluate': 0.5,
        'min_fit_clients': 2,
        'min_evaluate_clients': 2,
        'min_available_clients': 2,
        'batch_size': 32,
        'local_epochs': 1,
        'learning_rate': 0.01,
        'model_save_path': './saved_models',
        'save_interval': 5,
        'round_timeout': 600,
        'defense': 'none',
        'trim_fraction': 0.2,
        'num_malicious': 1
    }

    # 若指定配置文件，以配置文件覆盖默认值
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            # 兼容 GUI 写入的字段名（min_fit_clients / model_save_path）
            config.update(file_config)
            logger.info(f"已加载配置文件: {args.config}")
        except Exception as e:
            logger.warning(f"配置文件加载失败，使用默认配置: {e}")

    # 命令行参数最高优先级
    if args.port is not None:
        config['port'] = args.port
    if args.num_rounds is not None:
        config['num_rounds'] = args.num_rounds
    if args.min_clients is not None:
        config['min_fit_clients'] = args.min_clients
        config['min_evaluate_clients'] = args.min_clients
        config['min_available_clients'] = args.min_clients
    if args.local_epochs is not None:
        config['local_epochs'] = args.local_epochs
    if args.model_save_path is not None:
        config['model_save_path'] = args.model_save_path
    if args.defense is not None:
        config['defense'] = args.defense
    if args.trim_fraction is not None:
        config['trim_fraction'] = args.trim_fraction
    if args.num_malicious is not None:
        config['num_malicious'] = args.num_malicious

    # 启动服务器
    success, message = start_server(config)
    print(message)


if __name__ == "__main__":
    main()
