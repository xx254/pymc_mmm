#!/usr/bin/env python3
"""
测试中文字体修复
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import platform

def setup_chinese_font():
    """设置matplotlib中文字体支持"""
    # 根据系统选择合适的中文字体
    system = platform.system()
    
    if system == "Darwin":  # macOS
        font_candidates = [
            'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS',
            'SimHei', 'Microsoft YaHei', 'DejaVu Sans'
        ]
    elif system == "Windows":
        font_candidates = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
            'DejaVu Sans'
        ]
    else:  # Linux
        font_candidates = [
            'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback',
            'DejaVu Sans'
        ]
    
    # 查找可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(f"使用字体: {selected_font}")
        return True
    else:
        print("警告: 未找到合适的中文字体，将使用英文标签")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False

def test_font_display():
    """测试字体显示效果"""
    chinese_font_available = setup_chinese_font()
    
    # 创建测试数据
    dates = pd.date_range('2022-01-01', periods=100, freq='D')
    data1 = np.random.randn(100).cumsum()
    data2 = np.random.randn(100).cumsum()
    
    # 创建测试图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 标签字典
    labels = {
        'market_growth': '市场增长趋势' if chinese_font_available else 'Market Growth Trend',
        'marketing_channels': '营销渠道' if chinese_font_available else 'Marketing Channels',
        'x1_social': 'X1 (社交媒体)' if chinese_font_available else 'X1 (Social Media)',
        'x2_search': 'X2 (搜索引擎)' if chinese_font_available else 'X2 (Search Engine)',
        'sales': '销售额' if chinese_font_available else 'Sales',
        'date': '日期' if chinese_font_available else 'Date',
        'exposure': '曝光量' if chinese_font_available else 'Exposure',
        'competitor_offers': '竞争对手优惠' if chinese_font_available else 'Competitor Offers',
    }
    
    # 绘制测试图表
    axes[0, 0].plot(dates, data1, color='blue')
    axes[0, 0].set_title(labels['market_growth'])
    axes[0, 0].set_ylabel(labels['sales'])
    axes[0, 0].set_xlabel(labels['date'])
    
    axes[0, 1].plot(dates, data1, label=labels['x1_social'], color='blue')
    axes[0, 1].plot(dates, data2, label=labels['x2_search'], color='purple')
    axes[0, 1].set_title(labels['marketing_channels'])
    axes[0, 1].set_ylabel(labels['exposure'])
    axes[0, 1].set_xlabel(labels['date'])
    axes[0, 1].legend()
    
    axes[1, 0].plot(dates, -data1, color='orange')
    axes[1, 0].set_title(labels['competitor_offers'])
    axes[1, 0].set_ylabel(labels['competitor_offers'])
    axes[1, 0].set_xlabel(labels['date'])
    
    axes[1, 1].plot(dates, data1 + data2, color='black')
    axes[1, 1].set_title(labels['sales'])
    axes[1, 1].set_ylabel(labels['sales'])
    axes[1, 1].set_xlabel(labels['date'])
    
    plt.tight_layout()
    plt.savefig('font_test_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n测试完成！")
    print("如果图表中的中文正确显示，说明字体修复成功。")
    print("如果显示为方块，说明需要安装中文字体或使用英文标签。")
    print("图表已保存为 'font_test_result.png'")

if __name__ == "__main__":
    print("测试中文字体显示...")
    test_font_display() 