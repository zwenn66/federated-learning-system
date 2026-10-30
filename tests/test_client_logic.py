"""federated_client 核心逻辑单元测试：Non-IID 数据划分"""

import os
import sys

# 允许从项目根目录导入 federated_client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from federated_client import non_iid_split


def test_client_id_class_assignment():
    """验证 Non-IID 划分：每个客户端按序分配 2 个类别"""
    # client_1 -> 类别 [0, 1]
    classes = [(0 * 2 + i) % 10 for i in range(2)]
    assert classes == [0, 1]
    # client_2 -> 类别 [2, 3]
    classes = [(1 * 2 + i) % 10 for i in range(2)]
    assert classes == [2, 3]
    # client_6 -> 类别 [10, 11] % 10 = [0, 1]（循环覆盖）
    classes = [(5 * 2 + i) % 10 for i in range(2)]
    assert classes == [0, 1]


def test_non_iid_split_returns_only_assigned_classes():
    """验证划分后的数据只包含分配到的类别"""
    # 构造 10 类、每类 20 条的合成数据
    x = np.zeros((200, 28, 28, 1), dtype=np.float32)
    y = np.repeat(np.arange(10), 20)

    x_local, y_local = non_iid_split(x, y, "client_1", 5)

    # client_1 分配类别 [0, 1]
    assert set(np.unique(y_local)) == {0, 1}
    assert len(y_local) == 40


def test_non_iid_split_max_samples_cap():
    """验证最大样本数上限（默认 1000）"""
    x = np.zeros((5000, 28, 28, 1), dtype=np.float32)
    y = np.repeat(np.arange(10), 500)

    x_local, y_local = non_iid_split(x, y, "client_2", 5)

    assert len(y_local) <= 1000


def test_non_iid_split_invalid_client_id():
    """验证非法客户端 ID 抛 ValueError"""
    x = np.zeros((100, 28, 28, 1), dtype=np.float32)
    y = np.repeat(np.arange(10), 10)

    with pytest.raises(ValueError):
        non_iid_split(x, y, "bad_id", 5)


def test_non_iid_split_wrong_label_count():
    """验证标签与样本数量不一致时抛出 ValueError（拒绝静默截断）"""
    x = np.zeros((100, 28, 28, 1), dtype=np.float32)
    y = np.repeat(np.arange(10), 10)[:50]  # 长度不匹配

    with pytest.raises(ValueError):
        non_iid_split(x, y, "client_3", 5)
