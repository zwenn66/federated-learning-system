#!/usr/bin/env python3
"""
修复Matplotlib中文字体显示问题
"""

import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path


def check_chinese_font():
    """检查是否支持中文字体"""
    fonts = fm.findSystemFonts()
    chinese_fonts = []

    # 常见中文字体名称
    chinese_font_names = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong',
                          'SimSun', 'NSimSun', 'STHeiti', 'STKaiti', 'STSong',
                          'STFangsong', 'PingFang', 'Hiragino']

    for font in fonts:
        try:
            font_prop = fm.FontProperties(fname=font)
            font_name = font_prop.get_name()

            # 检查是否中文字体
            if any(chinese_name in font_name for chinese_name in chinese_font_names):
                chinese_fonts.append((font, font_name))
        except:
            continue

    return chinese_fonts


def fix_font_for_gui():
    """为GUI修复字体显示"""
    try:
        # 方法1：直接设置Matplotlib的字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi',
                                           'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 方法2：清除Matplotlib缓存
        cache_dir = matplotlib.get_cachedir()
        if os.path.exists(cache_dir):
            print(f"清除Matplotlib缓存: {cache_dir}")
            # 清除缓存文件
            for file in os.listdir(cache_dir):
                if file.endswith('.json') or file.endswith('.cache'):
                    try:
                        os.remove(os.path.join(cache_dir, file))
                    except:
                        pass

        print("字体设置已更新")
        return True

    except Exception as e:
        print(f"修复字体失败: {e}")
        return False


def test_chinese_display():
    """测试中文字体显示"""
    print("测试中文字体显示...")

    # 创建测试图表
    fig, ax = plt.subplots(figsize=(8, 6))

    # 设置标题和标签
    ax.set_title('中文字体测试 - 准确率变化')
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('准确率 (%)')

    # 模拟数据
    x = [1, 2, 3, 4, 5]
    y = [20, 35, 50, 65, 80]
    ax.plot(x, y, 'b-', marker='o', linewidth=2)
    ax.grid(True, alpha=0.3)

    # 添加中文图例
    ax.legend(['训练准确率'], loc='lower right')

    # 保存测试图表
    plt.tight_layout()
    plt.savefig('font_test.png', dpi=100)
    plt.show()

    print("测试图表已保存为 'font_test.png'")
    print("请查看图表标题和标签是否正常显示中文")


if __name__ == "__main__":
    print("=" * 60)
    print("中文字体修复工具")
    print("=" * 60)

    # 1. 检查当前字体
    chinese_fonts = check_chinese_font()
    if chinese_fonts:
        print(f"找到 {len(chinese_fonts)} 个中文字体:")
        for font_path, font_name in chinese_fonts[:5]:
            print(f"  - {font_name}: {font_path}")
    else:
        print("未找到中文字体")

    # 2. 修复字体设置
    print("\n修复字体设置...")
    fix_font_for_gui()

    # 3. 测试显示
    print("\n进行字体测试...")
    test_chinese_display()
