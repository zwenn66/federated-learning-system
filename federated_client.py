#!/usr/bin/env python3
import os
import json
import time
from typing import Tuple
import flwr as fl
import tensorflow as tf
import numpy as np
import argparse
import random

# 设置随机种子（保证实验可复现）
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def _log_local_metric(client_id: str, kind: str, samples: int, acc: float):
    """记录客户端本地训练指标到 client_logs/，供 GUI 客户端状态页读取

    kind: fit（本地训练）/ evaluate（全局评估）
    每轮追加一行 JSON，GUI 读取最新一条显示样本数与准确率。
    """
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'client_logs')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f'{client_id}.log'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "t": time.time(),
                "kind": kind,
                "samples": int(samples),
                "acc": round(float(acc), 4),
            }) + "\n")
    except Exception:
        pass  # 指标日志失败不影响训练


def load_fashion_mnist(client_id: str, num_clients: int) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    加载Fashion-MNIST数据集并完成预处理
    返回：(本地训练集, 全局测试集)
    """
    # 加载Fashion-MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # 数据预处理：归一化+添加通道维度（适配CNN输入）
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = np.expand_dims(x_train, axis=-1)  # (60000,28,28) → (60000,28,28,1)
    x_test = np.expand_dims(x_test, axis=-1)  # (10000,28,28) → (10000,28,28,1)

    # Non-IID划分：按客户端ID分配专属类别样本
    x_train_local, y_train_local = non_iid_split(x_train, y_train, client_id, num_clients)

    return (x_train_local, y_train_local), (x_test, y_test)


def non_iid_split(x: np.ndarray, y: np.ndarray, client_id: str, num_clients: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-IID划分策略：每个客户端分配2类样本（Fashion-MNIST共10类）
    """
    # 防御性校验：样本与标签数量必须一致，否则静默截断会掩盖数据问题
    if len(x) != len(y):
        raise ValueError(f"样本数量 {len(x)} 与标签数量 {len(y)} 不一致")
    try:
        # 提取客户端编号（如client_1 → 1 → 索引0）
        client_idx = int(client_id.split('_')[-1]) - 1
    except (IndexError, ValueError):
        raise ValueError(f"客户端ID格式错误，应为'client_数字'，当前：{client_id}")

    # 每客户端分配2类，循环覆盖10类
    classes_per_client = 2
    client_classes = [(client_idx * classes_per_client + i) % 10 for i in range(classes_per_client)]
    print(f"客户端 {client_id} 分配类别：{client_classes}")

    # 筛选当前客户端的样本
    mask = np.isin(y, client_classes)
    x_local = x[mask]
    y_local = y[mask]

    # 限制最大样本数（避免算力不足）
    max_samples = 1000
    if len(x_local) > max_samples:
        indices = np.random.choice(len(x_local), max_samples, replace=False)
        x_local = x_local[indices]
        y_local = y_local[indices]

    print(f"客户端 {client_id} 最终样本数：{len(x_local)}")
    return x_local, y_local


class FederatedClient(fl.client.NumPyClient):
    """符合Flower规范的联邦学习客户端类（仅定义一次）"""

    def __init__(self, client_id: str, num_clients: int):
        self.client_id = client_id
        self.num_clients = num_clients
        # 加载本地Non-IID数据和全局测试集
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_fashion_mnist(client_id, num_clients)
        # 初始化CNN模型（与服务器端保持一致）
        self.model = self._build_cnn_model()

    def _build_cnn_model(self) -> tf.keras.Model:
        """构建适配Fashion-MNIST的CNN模型"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')  # 10类服装分类
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',  # 标签为整数，无需one-hot
            metrics=['accuracy']
        )
        return model

    def get_parameters(self, config):
        """返回当前模型参数"""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """本地训练：使用Non-IID数据更新模型"""
        self.model.set_weights(parameters)
        # 从配置中获取本地训练轮数（由服务器指定）
        local_epochs = config.get('local_epochs', 1)
        # 本地训练
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=local_epochs,
            batch_size=32,
            verbose=0  # 静默模式，避免日志刷屏
        )
        # 记录本地训练指标（供 GUI 客户端状态页显示）
        train_acc = float(history.history['accuracy'][-1])
        _log_local_metric(self.client_id, "fit", len(self.x_train), train_acc)
        # 返回更新后的参数、样本数、训练指标
        return (
            self.model.get_weights(),
            len(self.x_train),
            {
                "train_loss": float(history.history['loss'][-1]),
                "train_acc": train_acc
            }
        )

    def evaluate(self, parameters, config):
        """模型评估：使用全局测试集验证泛化能力"""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        # 记录本地评估指标（供 GUI 客户端状态页显示）
        _log_local_metric(self.client_id, "evaluate", len(self.x_test), float(accuracy))
        # 返回损失、样本数、测试指标
        return float(loss), len(self.x_test), {"test_acc": float(accuracy)}

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='Fashion-MNIST联邦学习客户端')
    parser.add_argument('--cid', required=True, help='客户端ID，格式：client_1、client_2等')
    parser.add_argument('--server', required=True, help='服务器地址，如127.0.0.1:8080')
    parser.add_argument('--num-clients', type=int, default=3, help='客户端总数（用于Non-IID划分）')
    args = parser.parse_args()

    # 初始化客户端并连接服务器
    try:
        client = FederatedClient(args.cid, args.num_clients)
        print(f"\n客户端 {args.cid} 启动成功，连接服务器：{args.server}")
        fl.client.start_numpy_client(
            server_address=args.server,
            client=client,
            grpc_max_message_length=1024 * 1024 * 1024  # 增大消息长度，适配模型参数传输
        )
    except Exception as e:
        print(f"客户端启动失败：{str(e)}")
        exit(1)
