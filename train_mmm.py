import pandas as pd
import numpy as np
from pymc_marketing.mmm import (
    MMM,
    GeometricAdstock,
    LogisticSaturation,
)

# 读取数据
data = pd.read_csv('data/mmm_example.csv', parse_dates=['date_week'])

# 设置特征和目标变量
X = data.drop('y', axis=1)
y = data['y']

# 创建模型
mmm = MMM(
    date_column="date_week",
    channel_columns=["x1", "x2"],  # 营销渠道：社交媒体和搜索引擎
    adstock=GeometricAdstock(l_max=8),  # 使用几何衰减的滞后效应，最大滞后8期
    saturation=LogisticSaturation(),  # 使用逻辑函数建模饱和效应
    control_columns=[
        "event_1",  # 特殊事件1
        "event_2",  # 特殊事件2
        "t",        # 时间趋势
    ],
    yearly_seasonality=2,  # 使用2阶傅里叶级数建模年度季节性
)

# 设置采样参数
sampler_kwargs = {
    "draws": 1000,        # MCMC采样次数
    "tune": 1000,         # 预热期采样次数
    "target_accept": 0.9, # 目标接受率
    "chains": 4,          # MCMC链数
    "random_seed": 42,    # 随机种子
}

# 训练模型
print("开始训练模型...")
idata = mmm.fit(X, y, **sampler_kwargs)

# 保存模型
mmm.save("mmm_model.nc")
print("模型已保存到 mmm_model.nc")

# 打印模型评估指标
print("\n模型评估:")
print("="*50)
metrics = mmm.score(X, y)
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")

# 生成预测
print("\n生成预测...")
predictions = mmm.predict(X)
print("预测完成") 