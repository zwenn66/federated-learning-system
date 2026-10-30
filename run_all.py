#!/usr/bin/env python3
"""
修复版运行脚本 - 改进进程管理和错误处理
"""

import subprocess
import sys
import os
import time
import signal
import threading
import queue
from typing import List, Tuple
import argparse


class ProcessManager:
    """进程管理器"""

    def __init__(self):
        self.processes = []
        self.output_queue = queue.Queue()
        self.running = True

    def start_process(self, command: List[str], name: str, env=None) -> subprocess.Popen:
        """启动一个进程并捕获输出"""
        print(f"[启动] {name}...")

        # 创建进程
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env or os.environ
        )

        # 保存进程信息
        self.processes.append({
            'name': name,
            'process': process,
            'command': command
        })

        print(f"[成功] {name} 已启动 (PID: {process.pid})")

        # 启动输出捕获线程
        threading.Thread(
            target=self.capture_output,
            args=(name, process),
            daemon=True
        ).start()

        return process

    def capture_output(self, name: str, process: subprocess.Popen):
        """捕获进程输出"""

        # 捕获stdout
        def reader(pipe, queue, prefix):
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        queue.put(f"[{prefix}] {name}: {line.rstrip()}")
                pipe.close()
            except:
                pass

        # 启动读取线程
        threading.Thread(
            target=reader,
            args=(process.stdout, self.output_queue, "OUT"),
            daemon=True
        ).start()

        threading.Thread(
            target=reader,
            args=(process.stderr, self.output_queue, "ERR"),
            daemon=True
        ).start()

    def print_outputs(self):
        """打印输出队列中的内容"""
        try:
            while not self.output_queue.empty():
                print(self.output_queue.get_nowait())
        except queue.Empty:
            pass

    def stop_all(self):
        """停止所有进程"""
        print("\n[停止] 正在停止所有进程...")
        self.running = False

        # 先发送SIGTERM
        for proc_info in self.processes:
            proc = proc_info['process']
            if proc.poll() is None:  # 进程还在运行
                print(f"[停止] 正在停止 {proc_info['name']}...")
                try:
                    proc.terminate()
                except:
                    pass

        # 等待2秒
        time.sleep(2)

        # 强制杀掉还在运行的进程
        for proc_info in self.processes:
            proc = proc_info['process']
            if proc.poll() is None:  # 进程还在运行
                print(f"[强制停止] 强制停止 {proc_info['name']}...")
                try:
                    proc.kill()
                except:
                    pass

        # 等待进程结束
        for proc_info in self.processes:
            try:
                proc_info['process'].wait(timeout=1)
            except:
                pass

        print("[完成] 所有进程已停止")

    def check_all_running(self) -> bool:
        """检查所有进程是否在运行"""
        return all(p['process'].poll() is None for p in self.processes if p['name'] != "GUI")


def find_available_port(start_port: int = 8080) -> int:
    """寻找可用端口"""
    import socket
    for port in range(start_port, start_port + 100):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if result != 0:  # 端口可用
            return port
    return start_port


def main():
    parser = argparse.ArgumentParser(description='联邦学习系统启动脚本')
    parser.add_argument('--port', type=int, default=8080, help='服务器端口')
    parser.add_argument('--clients', type=int, default=2, help='客户端数量')
    parser.add_argument('--no-gui', action='store_true', help='不启动GUI')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    args = parser.parse_args()

    # 创建进程管理器
    manager = ProcessManager()

    try:
        # 1. 查找可用端口
        port = find_available_port(args.port)
        print(f"[配置] 使用端口: {port}")

        # 2. 启动服务器
        server_config = {
            'port': port,
            'num_rounds': 10,
            'min_fit_clients': max(2, args.clients),
            'local_epochs': 1,
            'model_save_path': './saved_models',
            'debug': args.debug
        }

        # 保存服务器配置
        import json
        with open('server_config_gui.json', 'w') as f:
            json.dump(server_config, f, indent=2)

        # 启动服务器进程（通过 --config 传入配置，与 GUI 启动方式一致）
        server_cmd = [
            sys.executable, 'federated_server.py',
            '--config', 'server_config_gui.json'
        ]

        server_env = os.environ.copy()
        if args.debug:
            server_env['TF_CPP_MIN_LOG_LEVEL'] = '0'  # TensorFlow详细日志
        else:
            server_env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow只显示错误

        manager.start_process(server_cmd, "服务器", server_env)

        # 等待服务器启动
        print("[等待] 等待服务器启动...")
        time.sleep(3)

        # 3. 启动客户端
        if not args.no_gui:
            print(f"[启动] 启动 {args.clients} 个客户端...")
            for i in range(args.clients):
                client_cmd = [
                    sys.executable, 'federated_client.py',
                    '--cid', f'client_{i + 1}',
                    '--server', f'127.0.0.1:{port}',
                    '--num-clients', str(args.clients)  # 补充num-clients参数，适配Non-IID划分
                ]
                manager.start_process(client_cmd, f"客户端_{i + 1}")
                time.sleep(1)  # 避免所有客户端同时连接

        # 4. 启动GUI（如果不禁止）
        if not args.no_gui:
            print("[启动] 启动GUI界面...")
            gui_cmd = [sys.executable, 'federated_gui.py']
            gui_process = manager.start_process(gui_cmd, "GUI")

            # 等待GUI进程
            print("[运行] 系统正在运行，按 Ctrl+C 停止")

            # 显示输出
            try:
                while manager.running:
                    manager.print_outputs()

                    # 检查GUI是否还在运行
                    if gui_process.poll() is not None:
                        print(f"[信息] GUI已退出，退出码: {gui_process.poll()}")
                        break

                    # 检查其他关键进程
                    if not manager.check_all_running():
                        print("[警告] 某些关键进程已停止")

                    time.sleep(1)

            except KeyboardInterrupt:
                print("\n[中断] 收到停止信号")

        else:
            # 没有GUI，等待用户中断
            print("[运行] 系统正在运行（无GUI），按 Ctrl+C 停止")
            print("-" * 50)
            print("服务器日志:")
            print("-" * 50)

            try:
                while manager.running:
                    manager.print_outputs()
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[中断] 收到停止信号")

    except Exception as e:
        print(f"[错误] 运行出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 停止所有进程
        manager.stop_all()


def alternative_main():
    """
    备选方案：更简单的启动方式
    分别启动各个组件，不在同一个脚本中管理
    """
    print("=" * 60)
    print("联邦学习系统启动指南")
    print("=" * 60)
    print()
    print("请按以下顺序分别在单独的终端中启动：")
    print()
    print("1. 启动服务器:")
    print("   python federated_server.py")
    print()
    print("2. 启动客户端（开多个终端）:")
    print("   python federated_client.py --cid client_1 --server 127.0.0.1:8080")
    print("   python federated_client.py --cid client_2 --server 127.0.0.1:8080")
    print()
    print("3. 启动GUI界面:")
    print("   python federated_gui.py")
    print()
    print("=" * 60)
    print("或者使用系统自带的终端，分别运行以上命令")
    print("=" * 60)


if __name__ == "__main__":
    # 检查必要的文件是否存在
    required_files = ['federated_server.py', 'federated_client.py', 'federated_gui.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"[错误] 缺少必要文件: {missing_files}")
        print("请确保所有文件都在同一目录下")
        sys.exit(1)

    print("联邦学习系统启动器")
    print("1. 自动启动所有组件")
    print("2. 查看手动启动指南")
    print("3. 仅启动服务器")

    try:
        choice = input("请选择 (1/2/3): ").strip()

        if choice == "1":
            main()
        elif choice == "2":
            alternative_main()
        elif choice == "3":
            # 仅启动服务器
            subprocess.run([sys.executable, 'federated_server.py'])
        else:
            print("无效选择，退出")

    except KeyboardInterrupt:
        print("\n用户中断，退出")
    except Exception as e:
        print(f"运行出错: {e}")
