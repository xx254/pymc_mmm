#!/usr/bin/env python3
"""
因果关系媒体混合建模教程
Understanding Causal Relationships in Media Mix Modeling

本教程演示如何在媒体混合建模中理解和应用因果关系，包括：
1. 因果识别的概念和重要性
2. 因果有向无环图(DAG)的构建
3. 数据生成过程模拟
4. 不同因果模型的比较
5. 使用高斯过程处理隐藏变量
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from graphviz import Digraph
import os
import sys

# 尝试导入seaborn，如果失败则使用matplotlib替代
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("警告: seaborn未安装，将使用matplotlib基础绘图")
    SEABORN_AVAILABLE = False
    
# 尝试导入IPython.display，如果失败则创建替代函数
try:
    from IPython.display import SVG, display
    IPYTHON_AVAILABLE = True
except ImportError:
    print("警告: IPython未安装，将跳过内联显示")
    IPYTHON_AVAILABLE = False
    def display(x):
        pass
    def SVG(x):
        return x

# 确保所有必要的包都能正确导入
try:
    import preliz as pz
    from pymc_marketing.mmm import MMM, GeometricAdstock, MichaelisMentenSaturation
    from pymc_marketing.mmm.transformers import geometric_adstock, michaelis_menten
    from pymc_marketing.prior import Prior
    PYMC_MARKETING_AVAILABLE = True
except ImportError as e:
    print(f"警告: PyMC-Marketing导入失败: {e}")
    print("请安装 pymc-marketing: pip install pymc-marketing")
    PYMC_MARKETING_AVAILABLE = False

# 设置绘图样式
plt.style.use('default')
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams.update({"figure.constrained_layout.use": True})

# 设置中文字体支持
def setup_chinese_font():
    """设置matplotlib中文字体支持"""
    import matplotlib.font_manager as fm
    import platform
    
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
    else:
        print("警告: 未找到合适的中文字体，将使用英文标签")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False
    
    return True

# 设置中文字体
CHINESE_FONT_AVAILABLE = setup_chinese_font()

# 标签字典 - 中文/英文标签对照
LABELS = {
    'title': {
        'market_growth': '市场增长趋势' if CHINESE_FONT_AVAILABLE else 'Market Growth Trend',
        'holiday_effect': '假期效应' if CHINESE_FONT_AVAILABLE else 'Holiday Effect',
        'competitor_offers': '竞争对手优惠' if CHINESE_FONT_AVAILABLE else 'Competitor Offers',
        'marketing_channels': '营销渠道' if CHINESE_FONT_AVAILABLE else 'Marketing Channels',
        'transformed_channels': '变换后的营销渠道' if CHINESE_FONT_AVAILABLE else 'Transformed Marketing Channels',
        'target_variable': '销售额 (目标变量)' if CHINESE_FONT_AVAILABLE else 'Sales (Target Variable)',
        'x1_recovery_corr': 'X1 渠道贡献恢复 - 相关性模型' if CHINESE_FONT_AVAILABLE else 'X1 Channel Contribution Recovery - Correlational Model',
        'x1_recovery_causal': 'X1 渠道贡献恢复 - 因果模型' if CHINESE_FONT_AVAILABLE else 'X1 Channel Contribution Recovery - Causal Model',
        'x2_recovery_corr': 'X2 渠道贡献恢复 - 相关性模型' if CHINESE_FONT_AVAILABLE else 'X2 Channel Contribution Recovery - Correlational Model',
        'x2_recovery_causal': 'X2 渠道贡献恢复 - 因果模型' if CHINESE_FONT_AVAILABLE else 'X2 Channel Contribution Recovery - Causal Model',
        'time_varying_intercept': '时变截距恢复效果' if CHINESE_FONT_AVAILABLE else 'Time-Varying Intercept Recovery',
    },
    'label': {
        'x1_social': 'X1 (社交媒体)' if CHINESE_FONT_AVAILABLE else 'X1 (Social Media)',
        'x2_search': 'X2 (搜索引擎)' if CHINESE_FONT_AVAILABLE else 'X2 (Search Engine)',
        'x1_transformed': 'X1 变换后' if CHINESE_FONT_AVAILABLE else 'X1 Transformed',
        'x2_transformed': 'X2 变换后' if CHINESE_FONT_AVAILABLE else 'X2 Transformed',
        'correlational_model': '相关性模型' if CHINESE_FONT_AVAILABLE else 'Correlational Model',
        'causal_model': '因果模型' if CHINESE_FONT_AVAILABLE else 'Causal Model',
        'true_effect': '真实效果' if CHINESE_FONT_AVAILABLE else 'True Effect',
        'true_hidden_factors': '真实隐藏因子 (截距 + 市场增长 - 竞争对手)' if CHINESE_FONT_AVAILABLE else 'True Hidden Factors (Intercept + Market Growth - Competitor)',
        'recovered_intercept': '恢复的时变截距' if CHINESE_FONT_AVAILABLE else 'Recovered Time-Varying Intercept',
        'confidence_interval': '95% 置信区间' if CHINESE_FONT_AVAILABLE else '95% Confidence Interval',
    },
    'axis': {
        'market_growth': '市场增长' if CHINESE_FONT_AVAILABLE else 'Market Growth',
        'holiday_signal': '假期信号' if CHINESE_FONT_AVAILABLE else 'Holiday Signal',
        'competitor_offers': '竞争对手优惠' if CHINESE_FONT_AVAILABLE else 'Competitor Offers',
        'exposure': '曝光量' if CHINESE_FONT_AVAILABLE else 'Exposure',
        'transformed_exposure': '变换后曝光量' if CHINESE_FONT_AVAILABLE else 'Transformed Exposure',
        'sales': '销售额' if CHINESE_FONT_AVAILABLE else 'Sales',
        'date': '日期' if CHINESE_FONT_AVAILABLE else 'Date',
        'intercept_value': '截距值' if CHINESE_FONT_AVAILABLE else 'Intercept Value',
    }
}

# 设置随机种子确保结果可重现
seed = sum(map(ord, "Causal MMM"))
rng = np.random.default_rng(seed)

class CausalMMMTutorial:
    """因果关系媒体混合建模教程类"""
    
    def __init__(self):
        self.seed = seed
        self.rng = rng
        self.sample_kwargs = {"draws": 500, "chains": 4, "nuts_sampler": "numpyro"}
        self.df = None
        self.data = None
        self.date_range = None
        
    def explain_causal_concepts(self):
        """解释因果识别的核心概念"""
        print("="*80)
        print("因果关系媒体混合建模教程")
        print("="*80)
        print()
        print("1. 什么是因果识别？")
        print("因果识别是确定我们是否可以使用现有数据和假设来证明因果关系的过程。")
        print("它帮助我们建立不同因素之间的明确联系，而不仅仅是观察它们的相关性。")
        print()
        print("2. 为什么在回归中理解因果关系很重要？")
        print("- 混淆偏差: 隐藏因素同时影响预测变量和结果变量")
        print("- 选择偏差: 非随机样本可能扭曲估计的关系")
        print("- 过度控制: 调整受治疗影响的变量可能导致错误的因果效应估计")
        print()
        print("3. 关键概念:")
        print("- 因果有向无环图(DAG): 显示假设因果关系的可视化工具")
        print("- 后门准则: 识别哪些变量可以阻断产生误导性连接的路径")
        print("- 最小调整集: 满足后门准则所需的最小变量组")
        print()
        
    def create_business_scenario_dag(self):
        """创建业务场景的因果图"""
        print("4. 业务场景:")
        print("假设您经营一家零售公司，在假期期间销售额增长。")
        print("您不是唯一的广告商，竞争对手也在推广他们的产品。")
        print()
        print("变量说明:")
        print("- Christmas (C): 假期季节提升消费者兴趣")
        print("- X1: 社交媒体广告 (Facebook, TikTok)")
        print("- X2: 搜索引擎广告")
        print("- Target (T): 销售收入")
        print("- Competitor Offers (I): 竞争对手优惠")
        print("- Market Growth (G): 市场增长")
        print()
        
        # 创建因果图
        dot = Digraph(comment='Business Scenario DAG')
        
        # 添加节点
        dot.node("C", "Christmas", style="dashed")
        dot.node("X1", "Marketing X1")
        dot.node("X2", "Marketing X2")
        dot.node("I", "Competitor Offers", style="dashed")
        dot.node("G", "Market Growth", style="dashed")
        dot.node("T", "Target")
        
        # 添加边
        dot.edge("C", "X1", style="dashed")
        dot.edge("C", "X2", style="dashed")
        dot.edge("I", "X2", style="dashed")
        dot.edge("X1", "X2")
        dot.edge("C", "T", style="dashed")
        dot.edge("X1", "T")
        dot.edge("X2", "T")
        dot.edge("I", "T", style="dashed")
        dot.edge("G", "T", style="dashed")
        
        # 保存图形
        try:
            dot.render('business_scenario_dag', format='png', cleanup=True)
            print("因果图已保存为 'business_scenario_dag.png'")
        except Exception as e:
            print(f"保存因果图时出错: {e}")
            
        return dot
        
    def generate_synthetic_data(self):
        """生成合成数据"""
        print("\n5. 数据生成过程:")
        print("我们将创建一个模拟数据集，反映上述因果关系...")
        
        # 创建日期范围
        min_date = pd.to_datetime("2022-01-01")
        max_date = pd.to_datetime("2024-11-06")
        self.date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        
        self.df = pd.DataFrame(data={"date_week": self.date_range}).assign(
            year=lambda x: x["date_week"].dt.year,
            month=lambda x: x["date_week"].dt.month,
            dayofyear=lambda x: x["date_week"].dt.dayofyear,
        )
        
        n = self.df.shape[0]
        print(f"观测数量: {n}")
        
        # 生成市场增长趋势
        self.df["market_growth"] = (np.linspace(start=0.0, stop=50, num=n) + 10) ** (1 / 4) - 1
        
        # 生成假期效应
        self._generate_holiday_effect(n)
        
        # 生成竞争对手优惠
        self._generate_competitor_offers(n)
        
        # 生成营销渠道数据
        self._generate_marketing_channels(n)
        
        # 应用adstock和饱和度变换
        self._apply_transformations()
        
        # 生成目标变量
        self._generate_target_variable(n)
        
        print("数据生成完成！")
        return self.df
        
    def _generate_holiday_effect(self, n):
        """生成假期效应"""
        holiday_dates = ["24-12"]  # 圣诞节
        std_devs = [25]
        holidays_coefficients = [2]
        
        holiday_signal = np.zeros(len(self.date_range))
        holiday_contributions = np.zeros(len(self.date_range))
        
        for holiday, std_dev, holiday_coef in zip(holiday_dates, std_devs, holidays_coefficients):
            holiday_occurrences = self.date_range[self.date_range.strftime("%d-%m") == holiday]
            
            for occurrence in holiday_occurrences:
                time_diff = (self.date_range - occurrence).days
                _holiday_signal = np.exp(-0.5 * (time_diff / std_dev) ** 2)
                holiday_signal += _holiday_signal
                holiday_contributions += _holiday_signal * holiday_coef
        
        self.df["holiday_signal"] = holiday_signal
        self.df["holiday_contributions"] = holiday_contributions
        
    def _generate_competitor_offers(self, n):
        """生成竞争对手优惠数据"""
        A = 0.5  # 振幅
        C = 2.5  # 中心
        omega = np.pi / (n / 2)
        
        self.df["competitor_offers"] = -A * np.cos(omega * self.df.index) + C
        
    def _generate_marketing_channels(self, n):
        """生成营销渠道数据"""
        # 生成X1 (社交媒体广告)
        try:
            x1 = np.random.normal(loc=5, scale=3, size=n)
        except:
            x1 = self.rng.normal(loc=5, scale=3, size=n)
            
        cofounder_effect_holiday_x1 = 2.5
        x1_conv = np.convolve(x1, np.ones(14) / 14, mode="same")
        
        # 处理边界效应
        noise = self.rng.normal(loc=0, scale=0.1, size=28)
        x1_conv[:14] = x1_conv.mean() + noise[:14]
        x1_conv[-14:] = x1_conv.mean() + noise[14:]
        
        self.df["x1"] = x1_conv + (self.df["holiday_signal"] * cofounder_effect_holiday_x1)
        
        # 生成X2 (搜索引擎广告)
        try:
            x2 = np.random.normal(loc=5, scale=2, size=n)
        except:
            x2 = self.rng.normal(loc=5, scale=2, size=n)
            
        cofounder_effect_holiday_x2 = 2.2
        cofounder_effect_x1_x2 = 1.3
        cofounder_effect_competitor_offers_x2 = -0.7
        
        x2_conv = np.convolve(x2, np.ones(18) / 12, mode="same")
        noise = self.rng.normal(loc=0, scale=0.1, size=28)
        x2_conv[:14] = x2_conv.mean() + noise[:14]
        x2_conv[-14:] = x2_conv.mean() + noise[14:]
        
        self.df["x2"] = (
            x2_conv
            + (self.df["holiday_signal"] * cofounder_effect_holiday_x2)
            + (self.df["x1"] * cofounder_effect_x1_x2)
            + (self.df["competitor_offers"] * cofounder_effect_competitor_offers_x2)
        )
        
    def _apply_transformations(self):
        """应用adstock和饱和度变换"""
        if not PYMC_MARKETING_AVAILABLE:
            print("跳过变换步骤 - PyMC-Marketing不可用")
            self.df["x1_adstock"] = self.df["x1"] * 0.8
            self.df["x2_adstock"] = self.df["x2"] * 0.6
            self.df["x1_adstock_saturated"] = self.df["x1_adstock"] * 0.9
            self.df["x2_adstock_saturated"] = self.df["x2_adstock"] * 0.9
            return
            
        # 应用几何adstock变换
        alpha1, alpha2 = 0.6, 0.2
        
        self.df["x1_adstock"] = (
            geometric_adstock(x=self.df["x1"].to_numpy(), alpha=alpha1, l_max=24, normalize=True)
            .eval().flatten()
        )
        
        self.df["x2_adstock"] = (
            geometric_adstock(x=self.df["x2"].to_numpy(), alpha=alpha2, l_max=24, normalize=True)
            .eval().flatten()
        )
        
        # 应用饱和度变换
        lam1, lam2 = 5.0, 9.0
        alpha_mm1, alpha_mm2 = 6, 12
        
        self.df["x1_adstock_saturated"] = michaelis_menten(
            x=self.df["x1_adstock"].to_numpy(), lam=lam1, alpha=alpha_mm1
        )
        
        self.df["x2_adstock_saturated"] = michaelis_menten(
            x=self.df["x2_adstock"].to_numpy(), lam=lam2, alpha=alpha_mm2
        )
        
    def _generate_target_variable(self, n):
        """生成目标变量"""
        self.df["intercept"] = 1.5
        self.df["epsilon"] = self.rng.normal(loc=0.0, scale=0.08, size=n)
        
        self.df["y"] = (
            self.df["intercept"]
            + self.df["market_growth"]
            - self.df["competitor_offers"]
            + self.df["holiday_contributions"]
            + self.df["x1_adstock_saturated"]
            + self.df["x2_adstock_saturated"]
            + self.df["epsilon"]
        )
        
        # 保存用于建模的数据
        columns_to_keep = ["date_week", "y", "x1", "x2"]
        self.data = self.df[columns_to_keep].copy()
        
    def plot_data_overview(self):
        """绘制数据概览"""
        if self.df is None:
            print("请先生成数据")
            return
            
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 市场增长趋势
        axes[0, 0].plot(self.df["date_week"], self.df["market_growth"], color="green")
        axes[0, 0].set_title(LABELS['title']['market_growth'])
        axes[0, 0].set_ylabel(LABELS['axis']['market_growth'])
        
        # 假期效应
        axes[0, 1].plot(self.df["date_week"], self.df["holiday_signal"], color="red")
        axes[0, 1].set_title(LABELS['title']['holiday_effect'])
        axes[0, 1].set_ylabel(LABELS['axis']['holiday_signal'])
        
        # 竞争对手优惠
        axes[1, 0].plot(self.df["date_week"], self.df["competitor_offers"], color="orange")
        axes[1, 0].set_title(LABELS['title']['competitor_offers'])
        axes[1, 0].set_ylabel(LABELS['axis']['competitor_offers'])
        
        # 营销渠道
        axes[1, 1].plot(self.df["date_week"], self.df["x1"], label=LABELS['label']['x1_social'], color="blue")
        axes[1, 1].plot(self.df["date_week"], self.df["x2"], label=LABELS['label']['x2_search'], color="purple")
        axes[1, 1].set_title(LABELS['title']['marketing_channels'])
        axes[1, 1].set_ylabel(LABELS['axis']['exposure'])
        axes[1, 1].legend()
        
        # 变换后的渠道
        axes[2, 0].plot(self.df["date_week"], self.df["x1_adstock_saturated"], label=LABELS['label']['x1_transformed'], color="blue")
        axes[2, 0].plot(self.df["date_week"], self.df["x2_adstock_saturated"], label=LABELS['label']['x2_transformed'], color="purple")
        axes[2, 0].set_title(LABELS['title']['transformed_channels'])
        axes[2, 0].set_ylabel(LABELS['axis']['transformed_exposure'])
        axes[2, 0].legend()
        
        # 目标变量
        axes[2, 1].plot(self.df["date_week"], self.df["y"], color="black")
        axes[2, 1].set_title(LABELS['title']['target_variable'])
        axes[2, 1].set_ylabel(LABELS['axis']['sales'])
        
        plt.tight_layout()
        plt.savefig("data_overview.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_correlational_model(self):
        """运行相关性模型"""
        if not PYMC_MARKETING_AVAILABLE:
            print("跳过相关性模型 - PyMC-Marketing不可用")
            return None
            
        print("\n6. 相关性模型:")
        print("首先，我们运行一个简单的相关性模型，不考虑因果关系...")
        
        X = self.data.drop("y", axis=1)
        y = self.data["y"]
        
        correlational_mmm = MMM(
            sampler_config=self.sample_kwargs,
            date_column="date_week",
            adstock=GeometricAdstock(l_max=24),
            saturation=MichaelisMentenSaturation(),
            channel_columns=["x1", "x2"],
        )
        
        try:
            correlational_mmm.fit(X=X, y=y, target_accept=0.90, random_seed=self.rng)
            correlational_mmm.sample_posterior_predictive(
                X, extend_idata=True, combined=True, random_seed=self.rng
            )
            
            # 检查分歧
            divergences = correlational_mmm.idata["sample_stats"]["diverging"].sum().item()
            print(f"相关性模型分歧数: {divergences}")
            
            return correlational_mmm
            
        except Exception as e:
            print(f"相关性模型训练失败: {e}")
            return None
            
    def create_causal_dag_string(self, version="simple"):
        """创建因果DAG字符串"""
        if version == "simple":
            return """
            digraph {
                x1 -> y;
                x2 -> y;
                holiday_signal -> y;
            }
            """
        elif version == "confounded":
            return """
            digraph {
                x1 -> y;
                x2 -> y;
                holiday_signal -> y;
                holiday_signal -> x1;
                holiday_signal -> x2;
            }
            """
        elif version == "full":
            return """
            digraph {
                x1 -> y;
                x2 -> y;
                x1 -> x2;
                holiday_signal -> y;
                holiday_signal -> x1;
                holiday_signal -> x2;
                competitor_offers -> x2;
                competitor_offers -> y;
                market_growth -> y;
            }
            """
        else:
            raise ValueError("版本必须是 'simple', 'confounded', 或 'full'")
            
    def run_causal_model(self, version="full"):
        """运行因果模型"""
        if not PYMC_MARKETING_AVAILABLE:
            print("跳过因果模型 - PyMC-Marketing不可用")
            return None
            
        print(f"\n7. 因果模型 ({version}):")
        print("现在我们运行考虑因果关系的模型...")
        
        # 准备数据
        if version in ["confounded", "full"]:
            data_with_controls = self.data.copy()
            data_with_controls["holiday_signal"] = self.df["holiday_signal"]
            X = data_with_controls.drop("y", axis=1)
            y = data_with_controls["y"]
            control_columns = ["holiday_signal"]
        else:
            X = self.data.drop("y", axis=1)
            y = self.data["y"]
            control_columns = None
            
        # 创建因果DAG
        causal_dag = self.create_causal_dag_string(version)
        
        # 创建模型
        model_config = {}
        if version == "full":
            # 使用时变截距处理未观测到的混淆因子
            causal_mmm = MMM(
                sampler_config=self.sample_kwargs,
                date_column="date_week",
                adstock=GeometricAdstock(l_max=24),
                saturation=MichaelisMentenSaturation(),
                channel_columns=["x1", "x2"],
                control_columns=control_columns,
                outcome_node="y",
                dag=causal_dag,
                time_varying_intercept=True,
            )
            causal_mmm.model_config["intercept_tvp_config"].ls_mu = 180
            causal_mmm.model_config["intercept"] = Prior("Normal", mu=1, sigma=2)
        else:
            causal_mmm = MMM(
                sampler_config=self.sample_kwargs,
                date_column="date_week",
                adstock=GeometricAdstock(l_max=24),
                saturation=MichaelisMentenSaturation(),
                channel_columns=["x1", "x2"],
                control_columns=control_columns,
                outcome_node="y",
                dag=causal_dag,
            )
        
        # 显示调整集
        print(f"调整集: {causal_mmm.causal_graphical_model.adjustment_set}")
        print(f"最小调整集: {causal_mmm.causal_graphical_model.minimal_adjustment_set}")
        
        try:
            causal_mmm.fit(X=X, y=y, target_accept=0.95, random_seed=self.rng)
            causal_mmm.sample_posterior_predictive(
                X, extend_idata=True, combined=True, random_seed=self.rng
            )
            
            # 检查分歧
            divergences = causal_mmm.idata["sample_stats"]["diverging"].sum().item()
            print(f"因果模型分歧数: {divergences}")
            
            return causal_mmm
            
        except Exception as e:
            print(f"因果模型训练失败: {e}")
            return None
            
    def compare_models(self, correlational_mmm, causal_mmm):
        """比较不同模型的效果"""
        if not PYMC_MARKETING_AVAILABLE or correlational_mmm is None or causal_mmm is None:
            print("跳过模型比较 - 模型不可用")
            return
            
        print("\n8. 模型比较:")
        
        # 计算R²分数
        r2_corr = az.r2_score(
            y_true=self.df["y"].values,
            y_pred=correlational_mmm.idata.posterior_predictive.stack(sample=("chain", "draw"))["y"].values.T
            * correlational_mmm.target_transformer["scaler"].scale_.item(),
        ).iloc[0]
        
        r2_causal = az.r2_score(
            y_true=self.df["y"].values,
            y_pred=causal_mmm.idata.posterior_predictive.stack(sample=("chain", "draw"))["y"].values.T
            * causal_mmm.target_transformer["scaler"].scale_.item(),
        ).iloc[0]
        
        print(f"相关性模型 R²: {r2_corr:.3f}")
        print(f"因果模型 R²: {r2_causal:.3f}")
        
        # 比较渠道贡献恢复效果
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 相关性模型恢复效果
        corr_contrib = (
            correlational_mmm.fit_result.channel_contribution.mean(dim=["chain", "draw"])
            * correlational_mmm.target_transformer["scaler"].scale_.item()
        )
        
        # 因果模型恢复效果  
        causal_contrib = (
            causal_mmm.fit_result.channel_contribution.mean(dim=["chain", "draw"])
            * causal_mmm.target_transformer["scaler"].scale_.item()
        )
        
        # X1 比较
        axes[0, 0].plot(self.date_range, corr_contrib.sel(channel="x1"), 
                       label=LABELS['label']['correlational_model'], linestyle="--", color="blue")
        axes[0, 0].plot(self.date_range, self.df["x1_adstock_saturated"], 
                       label=LABELS['label']['true_effect'], color="black")
        axes[0, 0].set_title(LABELS['title']['x1_recovery_corr'])
        axes[0, 0].legend()
        
        axes[0, 1].plot(self.date_range, causal_contrib.sel(channel="x1"), 
                       label=LABELS['label']['causal_model'], linestyle="--", color="blue")
        axes[0, 1].plot(self.date_range, self.df["x1_adstock_saturated"], 
                       label=LABELS['label']['true_effect'], color="black")
        axes[0, 1].set_title(LABELS['title']['x1_recovery_causal'])
        axes[0, 1].legend()
        
        # X2 比较
        axes[1, 0].plot(self.date_range, corr_contrib.sel(channel="x2"), 
                       label=LABELS['label']['correlational_model'], linestyle="--", color="orange")
        axes[1, 0].plot(self.date_range, self.df["x2_adstock_saturated"], 
                       label=LABELS['label']['true_effect'], color="black")
        axes[1, 0].set_title(LABELS['title']['x2_recovery_corr'])
        axes[1, 0].legend()
        
        axes[1, 1].plot(self.date_range, causal_contrib.sel(channel="x2"), 
                       label=LABELS['label']['causal_model'], linestyle="--", color="orange")
        axes[1, 1].plot(self.date_range, self.df["x2_adstock_saturated"], 
                       label=LABELS['label']['true_effect'], color="black")
        axes[1, 1].set_title(LABELS['title']['x2_recovery_causal'])
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_time_varying_intercept(self, causal_mmm):
        """绘制时变截距"""
        if not PYMC_MARKETING_AVAILABLE or causal_mmm is None:
            print("跳过时变截距绘图 - 模型不可用")
            return
            
        print("\n9. 时变截距分析:")
        print("时变截距可以捕捉未观测到的混淆因子...")
        
        # 恢复截距效果
        intercept_effect = (
            az.hdi(causal_mmm.fit_result["intercept"], hdi_prob=0.95)
            * causal_mmm.target_transformer["scaler"].scale_.item()
        )
        mean_intercept = (
            causal_mmm.fit_result.intercept.mean(dim=["chain", "draw"])
            * causal_mmm.target_transformer["scaler"].scale_.item()
        )
        
        # 真实的隐藏因子组合
        true_hidden = (
            self.df["intercept"] + self.df["market_growth"] - self.df["competitor_offers"]
        )
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.date_range, true_hidden, 
                label=LABELS['label']['true_hidden_factors'], color="black")
        plt.plot(self.date_range, mean_intercept, 
                label=LABELS['label']['recovered_intercept'], linestyle="--", color="red")
        plt.fill_between(
            self.date_range,
            intercept_effect.intercept.isel(hdi=0),
            intercept_effect.intercept.isel(hdi=1),
            alpha=0.2,
            color="red",
            label=LABELS['label']['confidence_interval']
        )
        plt.title(LABELS['title']['time_varying_intercept'])
        plt.xlabel(LABELS['axis']['date'])
        plt.ylabel(LABELS['axis']['intercept_value'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("time_varying_intercept.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_complete_tutorial(self):
        """运行完整教程"""
        print("开始因果关系媒体混合建模教程...")
        
        # 1. 解释概念
        self.explain_causal_concepts()
        
        # 2. 创建因果图
        self.create_business_scenario_dag()
        
        # 3. 生成数据
        self.generate_synthetic_data()
        
        # 4. 绘制数据概览
        self.plot_data_overview()
        
        # 5. 运行相关性模型
        correlational_mmm = self.run_correlational_model()
        
        # 6. 运行因果模型
        causal_mmm = self.run_causal_model(version="full")
        
        # 7. 比较模型
        self.compare_models(correlational_mmm, causal_mmm)
        
        # 8. 分析时变截距
        self.plot_time_varying_intercept(causal_mmm)
        
        print("\n="*80)
        print("教程总结:")
        print("="*80)
        print("1. 因果识别帮助我们理解真正的因果关系，而不仅仅是相关性")
        print("2. 正确的因果DAG可以指导我们选择合适的控制变量")
        print("3. 时变截距可以处理未观测到的混淆因子")
        print("4. 因果模型通常比纯相关性模型更能准确恢复真实效果")
        print("5. 理解业务环境和因果关系结构对于建立有效的MMM模型至关重要")
        print("\n教程完成！所有图表已保存到当前目录。")
        
        return {
            'correlational_model': correlational_mmm,
            'causal_model': causal_mmm,
            'data': self.data,
            'full_data': self.df
        }

def main():
    """主函数"""
    print("欢迎使用因果关系媒体混合建模教程！")
    print("本教程将演示如何在MMM中应用因果推理概念。")
    print()
    
    # 创建教程实例
    tutorial = CausalMMMTutorial()
    
    # 运行完整教程
    results = tutorial.run_complete_tutorial()
    
    return results

if __name__ == "__main__":
    results = main() 