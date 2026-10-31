#!/usr/bin/env python3
"""准备 Fashion-MNIST 数据集（TensorFlow/Keras 缓存格式）

背景：TensorFlow 2.16 的 tf.keras.datasets.fashion_mnist.load_data()
会从 Google 服务器下载 4 个 idx gz 文件，国内网络常被阻断/超时。
本脚本改从 hf-mirror.com（国内 HuggingFace 镜像）下载官方 parquet 数据，
再转换成 Keras 缓存所需的 idx gz 文件。

缓存位置：~/.keras/datasets/fashion-mnist/（4 个 idx gz，共约 30MB）

用法：
    .\.venv\Scripts\python scripts\fetch_fashion_mnist.py
"""

import gzip
import io
import os
import struct

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".keras", "datasets", "fashion-mnist")
HF_DIR = os.path.join(os.path.expanduser("~"), ".keras", "datasets", "hf_fashion")
HF_BASE = "https://hf-mirror.com/datasets/zalando-datasets/fashion_mnist/resolve/main/fashion_mnist"

PARQUET_FILES = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

TARGET_GZ = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _download(url: str, dest: str) -> bool:
    """下载文件（用 curl，兼容国内网络限制）"""
    import subprocess
    print(f"下载 {url}")
    result = subprocess.run(
        ["curl", "-L", "-s", "--max-time", "400", "-o", dest, url],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) < 100_000:
        print(f"  下载失败: {url}")
        return False
    print(f"  -> {os.path.getsize(dest)} bytes")
    return True


def _load_parquet(path):
    t = pq.read_table(path)
    n = t.num_rows
    images = t.column("image").combine_chunks().field("bytes")
    labels = t.column("label")

    x = np.zeros((n, 28, 28), dtype=np.uint8)
    y = np.zeros(n, dtype=np.uint8)

    idx = 0
    for b in images.to_pylist():
        img = Image.open(io.BytesIO(b)).convert("L")
        x[idx] = np.array(img)
        idx += 1
    y[:] = np.asarray(labels.to_pylist(), dtype=np.uint8)
    return x, y


def _write_idx_gz(path, header, data):
    with gzip.open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


def _build_idx_gz():
    """从 parquet 生成 Keras 缓存的 idx gz 文件"""
    os.makedirs(CACHE_DIR, exist_ok=True)

    for split in ("train", "test"):
        parquet_path = os.path.join(HF_DIR, PARQUET_FILES[split])
        if not os.path.exists(parquet_path):
            print(f"缺少 parquet 文件: {parquet_path}")
            return False
        print(f"解析 {split}.parquet ...")
        x, y = load_cache_or_parse(parquet_path, split)

        img_gz, label_gz = TARGET_GZ[split]
        _write_idx_gz(os.path.join(CACHE_DIR, img_gz),
                      struct.pack(">IIII", 2051, x.shape[0], 28, 28), x)
        _write_idx_gz(os.path.join(CACHE_DIR, label_gz),
                      struct.pack(">II", 2049, y.shape[0]), y)
        print(f"  ✅ {img_gz} ({os.path.getsize(os.path.join(CACHE_DIR, img_gz))} bytes)")

    return True


def load_cache_or_parse(parquet_path, split):
    """解析 parquet（train 60000 张较慢，解析后缓存中间结果）"""
    cache_npz = os.path.join(HF_DIR, f"{split}.npz")
    if os.path.exists(cache_npz):
        data = np.load(cache_npz)
        return data["x"], data["y"]
    x, y = _load_parquet(parquet_path)
    np.savez(cache_npz, x=x, y=y)
    return x, y


def main():
    # 检查是否已有完整数据（存在且大于 20MB 即视为有效）
    train_img = os.path.join(CACHE_DIR, "train-images-idx3-ubyte.gz")
    if os.path.exists(train_img) and os.path.getsize(train_img) > 20_000_000:
        print("✅ 数据已就绪，无需处理")
        return

    # 1. 下载 parquet
    os.makedirs(HF_DIR, exist_ok=True)
    for split, fname in PARQUET_FILES.items():
        dest = os.path.join(HF_DIR, fname)
        if not os.path.exists(dest) or os.path.getsize(dest) < 100_000:
            if not _download(f"{HF_BASE}/{fname}", dest):
                print("❌ 下载失败，请检查网络后重试")
                return

    # 2. 生成 idx gz
    if _build_idx_gz():
        print("✅ Fashion-MNIST 数据准备完成，可以直接运行训练了")


if __name__ == "__main__":
    main()
