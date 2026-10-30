#!/usr/bin/env python3
"""
使用阿里云镜像安装依赖包
"""

import subprocess
import sys


def install_packages():
    # 使用阿里云镜像
    mirror_url = "https://mirrors.aliyun.com/pypi/simple/"

    packages = [
        'tensorflow>=2.10.0',
        'flwr>=1.4.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'pillow>=9.0.0',
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0'
    ]

    print(f"使用阿里云镜像安装: {mirror_url}")
    print("正在安装依赖包...")

    # 先升级pip
    print("升级pip...")
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        '--upgrade', 'pip',
        '-i', mirror_url, '--trusted-host', 'mirrors.aliyun.com'
    ])

    # 安装所有包
    for package in packages:
        print(f"安装 {package}...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                '--upgrade', package,
                '-i', mirror_url, '--trusted-host', 'mirrors.aliyun.com',
                '--timeout', '100'
            ])
            print(f"  ✓ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {package} 安装失败: {e}")

    print("\n所有依赖包安装完成！")


if __name__ == "__main__":
    install_packages()
