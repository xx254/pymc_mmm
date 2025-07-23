import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('data/mmm_example.csv')

# 生成数据概览
print("\n数据基本信息:")
print("="*50)
print(f"总行数: {len(df)}")
print(f"时间范围: {df['date_week'].min()} 到 {df['date_week'].max()}")
print("\n各列统计信息:")
print("="*50)
print(df.describe())

print("\n各列数据类型:")
print("="*50)
print(df.dtypes)

# 生成示例数据
print("\n数据示例 (前5行):")
print("="*50)
print(df.head().to_string())

# 导出示例CSV格式
with open('data_format.csv', 'w') as f:
    f.write("date_week,y,x1,x2,event_1,event_2,dayofyear,t\n")
    f.write("2018-04-02,3984.66,0.32,0.00,0,0,92,0\n")
    f.write("2018-04-09,3762.87,0.11,0.00,0,0,99,1\n")
    f.write("...\n")
    f.write("2021-08-30,4675.97,0.44,0.00,0,0,242,178\n")

print("\n变量说明:")
print("="*50)
print("date_week  : 周数据的日期")
print("y          : 目标变量（销售额）")
print("x1         : 社交媒体投放强度 (0-1)")
print("x2         : 搜索引擎投放强度 (0-1)")
print("event_1    : 事件1标记 (0/1)")
print("event_2    : 事件2标记 (0/1)")
print("dayofyear  : 一年中的第几天 (1-365)")
print("t          : 时间序列索引 (从0开始)")

# 计算一些关键统计信息
print("\n关键统计:")
print("="*50)
print(f"平均销售额: {df['y'].mean():.2f}")
print(f"销售额标准差: {df['y'].std():.2f}")
print(f"社交媒体投放平均强度: {df['x1'].mean():.2f}")
print(f"搜索引擎投放平均强度: {df['x2'].mean():.2f}")
print(f"Event 1 发生次数: {df['event_1'].sum()}")
print(f"Event 2 发生次数: {df['event_2'].sum()}")

# 导出完整的数据格式说明
with open('data_format_description.txt', 'w') as f:
    f.write("""训练数据格式说明:

文件格式: CSV (Comma Separated Values)
时间频率: 周度数据
时间范围: 2018-04-02 到 2021-08-30

列说明:
1. date_week  : 周数据的日期 (YYYY-MM-DD格式)
2. y          : 目标变量（销售额）
                - 范围: 3000-8500
                - 均值: ~5300
3. x1         : 社交媒体投放强度
                - 范围: 0-1
                - 表示社交媒体营销活动的强度
4. x2         : 搜索引擎投放强度
                - 范围: 0-1
                - 多数时候为0
                - 投放时通常在0.8-1.0之间
5. event_1    : 事件1标记
                - 二元变量(0/1)
                - 标记特殊促销活动
6. event_2    : 事件2标记
                - 二元变量(0/1)
                - 标记特殊促销活动
7. dayofyear  : 一年中的第几天
                - 范围: 1-365
                - 用于捕捉季节性效应
8. t          : 时间序列索引
                - 从0开始的连续整数
                - 用于时间趋势建模

数据特点:
- 周度数据，共178周
- 包含两种营销渠道的投放强度
- 记录了两次特殊促销活动
- 包含季节性信息
- 适合因果推断和营销效果分析

使用建议:
1. 建议在建模前对销售额进行标准化
2. 注意处理季节性效应
3. 考虑事件对销售的影响
4. 可以利用dayofyear进行季节性分解
""") 