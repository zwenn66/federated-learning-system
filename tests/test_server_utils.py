"""federated_server 工具函数单元测试：端口检测"""

import os
import sys
import socket

# 允许从项目根目录导入 federated_server（不触发训练逻辑）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _get_random_free_port():
    """获取一个当前空闲的随机端口"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_check_port_available():
    """验证空闲端口被判定为可用"""
    from federated_server import check_port_available
    port = _get_random_free_port()
    assert check_port_available(port) is True


def test_check_port_occupied():
    """验证被占用端口被判定为不可用"""
    from federated_server import check_port_available
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    try:
        assert check_port_available(port) is False
    finally:
        sock.close()


def test_find_available_port():
    """验证 find_available_port 返回可用端口"""
    from federated_server import find_available_port, check_port_available
    start = _get_random_free_port()
    port = find_available_port(start_port=start, max_attempts=50)
    assert check_port_available(port) is True
