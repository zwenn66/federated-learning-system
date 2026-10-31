#!/usr/bin/env python3
"""联邦学习系统冒烟测试

一键验证「服务器 + N 客户端 + FedAvg 训练」全链路：
1. 以子进程启动服务器（2 轮、2 客户端）
2. 启动 2 个客户端
3. 等待训练完成，检查 server.log 中的轮次与指标
4. 检查全局模型文件是否生成
5. 自动清理所有进程

用法：
    .\.venv\Scripts\python scripts\smoke_test.py [--rounds 2] [--clients 2] [--port 8095]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOG_FILE = os.path.join(ROOT, 'server.log')
MODEL_DIR = os.path.join(ROOT, 'saved_models')


def main():
    parser = argparse.ArgumentParser(description='联邦学习系统冒烟测试')
    parser.add_argument('--rounds', type=int, default=2, help='训练轮数（默认2）')
    parser.add_argument('--clients', type=int, default=2, help='客户端数量（默认2）')
    parser.add_argument('--port', type=int, default=8095, help='服务器端口（默认8095）')
    parser.add_argument('--timeout', type=int, default=900, help='最长等待秒数（默认900）')
    args = parser.parse_args()

    # 清理旧日志，确保从干净状态开始
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    processes = []

    def stop_all():
        for p in processes:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        time.sleep(1)
        for p in processes:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

    print(f"[1/4] 启动服务器 (端口 {args.port}, {args.rounds} 轮, {args.clients} 客户端)...")
    # 注意：子进程输出重定向到 DEVNULL，避免 PIPE 缓冲区填满阻塞训练
    # （服务器日志已通过 logging FileHandler 写入 server.log）
    server = subprocess.Popen(
        [PY, 'federated_server.py',
         '--port', str(args.port),
         '--num-rounds', str(args.rounds),
         '--min-clients', str(args.clients),
         '--local-epochs', '1',
         '--model-save-path', './saved_models'],
        cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    processes.append(server)
    time.sleep(8)  # 等待服务器就绪（含全局模型初始化）

    print(f"[2/4] 启动 {args.clients} 个客户端...")
    clients = []
    for i in range(args.clients):
        c = subprocess.Popen(
            [PY, 'federated_client.py',
             '--cid', f'client_{i + 1}',
             '--server', f'127.0.0.1:{args.port}',
             '--num-clients', str(args.clients)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        clients.append(c)
        processes.append(c)
        time.sleep(2)

    print(f"[3/4] 等待训练完成（最长 {args.timeout}s，首次运行需下载 Fashion-MNIST 数据集）...")
    deadline = time.time() + args.timeout
    target_round = f'[ROUND {args.rounds}] FitMetrics'
    completed = False
    while time.time() < deadline:
        time.sleep(10)
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if target_round in content:
                completed = True
                break
        # 服务器进程提前退出 = 失败
        if server.poll() is not None:
            print("[失败] 服务器进程提前退出")
            # 输出服务器日志尾部帮助排查
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                    print(f.read()[-2000:])
            stop_all()
            sys.exit(1)

    print(f"[4/4] 检查结果...")
    if not completed:
        print("❌ 冒烟测试失败：超时未完成训练")
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                print("--- server.log 尾部 ---")
                print(f.read()[-3000:])
        stop_all()
        sys.exit(1)

    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    fit_rounds = re.findall(r'\[ROUND (\d+)\] FitMetrics = \{(.*?)\}', content)
    eval_rounds = re.findall(r'\[ROUND (\d+)\] EvaluateMetrics = \{(.*?)\}', content)
    print(f"✅ 训练链路完成，捕获 FitMetrics {len(fit_rounds)} 条 / EvaluateMetrics {len(eval_rounds)} 条")
    for r, m in fit_rounds:
        print(f"   ROUND {r} 训练: {m}")
    for r, m in eval_rounds:
        print(f"   ROUND {r} 评估: {m}")

    # 检查全局模型
    model_files = [f for f in os.listdir(MODEL_DIR)
                   if f.startswith('global_model_round') and f.endswith('.h5')]
    print(f"全局模型文件数: {len(model_files)}，最新: {sorted(model_files)[-1] if model_files else '无'}")

    stop_all()
    print("✅ 冒烟测试通过")


if __name__ == '__main__':
    main()
