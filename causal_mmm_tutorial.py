#!/usr/bin/env python3
"""
Causal Media Mix Modeling Tutorial
Understanding Causal Relationships in Media Mix Modeling

This tutorial demonstrates how to understand and apply causal relationships in media mix modeling, including:
1. The concept and importance of causal identification
2. Construction of causal directed acyclic graphs (DAGs)
3. Data generation process simulation
4. Comparison of different causal models
5. Using Gaussian processes to handle latent variables
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
import traceback

# Try to import seaborn, fall back to matplotlib if it fails
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("Warning: seaborn not installed, will use basic matplotlib plotting")
    SEABORN_AVAILABLE = False
    
# Try to import IPython.display, create fallback function if it fails
try:
    from IPython.display import SVG, display
    IPYTHON_AVAILABLE = True
except ImportError:
    print("Warning: IPython not installed, will skip inline display")
    IPYTHON_AVAILABLE = False
    def display(x):
        pass
    def SVG(x):
        return x

# Ensure all necessary packages can be imported correctly
try:
    import preliz as pz
    from pymc_marketing.mmm import MMM, GeometricAdstock, MichaelisMentenSaturation
    from pymc_marketing.mmm.transformers import geometric_adstock, michaelis_menten
    from pymc_marketing.prior import Prior
    PYMC_MARKETING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyMC-Marketing import failed: {e}")
    print("Please install pymc-marketing: pip install pymc-marketing")
    PYMC_MARKETING_AVAILABLE = False

# Set plotting style
plt.style.use('default')
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams.update({"figure.constrained_layout.use": True})

# Set Chinese font support
def setup_chinese_font():
    """Set up matplotlib Chinese font support"""
    import matplotlib.font_manager as fm
    import platform
    
    # Choose appropriate Chinese font based on system
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
    
    # Find available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue
        print(f"Using font: {selected_font}")
    else:
        print("Warning: No suitable Chinese font found, will use English labels")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False
    
    return True

# Set English font (disable Chinese font support)
CHINESE_FONT_AVAILABLE = False

# Label dictionary - use English labels only
LABELS = {
    'title': {
        'market_growth': 'Market Growth Trend',
        'holiday_effect': 'Holiday Effect',
        'competitor_offers': 'Competitor Offers',
        'marketing_channels': 'Marketing Channels',
        'transformed_channels': 'Transformed Marketing Channels',
        'target_variable': 'Sales (Target Variable)',
        'x1_recovery_corr': 'X1 Channel Contribution Recovery - Correlational Model',
        'x1_recovery_causal': 'X1 Channel Contribution Recovery - Causal Model',
        'x2_recovery_corr': 'X2 Channel Contribution Recovery - Correlational Model',
        'x2_recovery_causal': 'X2 Channel Contribution Recovery - Causal Model',
        'time_varying_intercept': 'Time-Varying Intercept Recovery',
    },
    'label': {
        'x1_social': 'X1 (Social Media)',
        'x2_search': 'X2 (Search Engine)',
        'x1_transformed': 'X1 Transformed',
        'x2_transformed': 'X2 Transformed',
        'correlational_model': 'Correlational Model',
        'causal_model': 'Causal Model',
        'true_effect': 'True Effect',
        'true_hidden_factors': 'True Hidden Factors (Intercept + Market Growth - Competitor)',
        'recovered_intercept': 'Recovered Time-Varying Intercept',
        'confidence_interval': '95% Confidence Interval',
    },
    'axis': {
        'market_growth': 'Market Growth',
        'holiday_signal': 'Holiday Signal',
        'competitor_offers': 'Competitor Offers',
        'exposure': 'Exposure',
        'transformed_exposure': 'Transformed Exposure',
        'sales': 'Sales',
        'date': 'Date',
        'intercept_value': 'Intercept Value',
    }
}

# Set random seed to ensure reproducible results
seed = sum(map(ord, "Causal MMM"))
rng = np.random.default_rng(seed)

class CausalMMMTutorial:
    """Causal Media Mix Modeling Tutorial Class"""
    
    def __init__(self):
        """Initialize tutorial"""
        self.df = None
        self.data = None
        self.date_range = None
        self.rng = 42  # 使用简单的整数作为种子而不是生成器对象
        
        # 设置采样参数
        self.sample_kwargs = {
            "draws": 1000,       # 增加采样次数以提高有效样本量
            "tune": 1000,        # 增加调优步数以提高收敛性
            "chains": 4,         # 保持4条链以确保收敛诊断
            "nuts_sampler": "numpyro",  # 使用更快的numpyro采样器
            "target_accept": 0.9,  # 稍微降低目标接受率以加快收敛
            "return_inferencedata": True,
            "random_seed": 42    # 使用整数种子
        }
        
        # 尝试加载真实数据
        try:
            self.load_real_mmm_data()
        except Exception as e:
            print(f"Warning: Could not load real data: {e}")
            print("Will generate synthetic data instead")
            self.df = None
            self.data = None
    
    def explain_causal_concepts(self):
        """Explain the core concepts of causal identification"""
        print("="*80)
        print("Causal Media Mix Modeling Tutorial")
        print("="*80)
        print()
        print("1. What is causal identification?")
        print("Causal identification is the process of determining whether we can use existing data and assumptions to prove causal relationships.")
        print("It helps us establish clear connections between different factors, not just observe their correlations.")
        print()
        print("2. Why is understanding causality important in regression?")
        print("- Confounding bias: Hidden factors simultaneously affect predictors and outcome variables")
        print("- Selection bias: Non-random samples may distort estimated relationships")
        print("- Over-control: Adjusting for variables affected by treatment may lead to incorrect causal effect estimates")
        print()
        print("3. Key concepts:")
        print("- Causal Directed Acyclic Graph (DAG): Visualization tool showing assumed causal relationships")
        print("- Backdoor criterion: Identifies which variables can block paths that create misleading connections")
        print("- Minimal adjustment set: Minimal set of variables needed to satisfy the backdoor criterion")
        print()
        
    def create_business_scenario_dag(self):
        """Create causal graph for business scenario"""
        print("4. Business scenario:")
        print("Suppose you run a retail company with increased sales during holidays.")
        print("You're not the only advertiser; competitors are also promoting their products.")
        print()
        print("Variable descriptions:")
        print("- Christmas (C): Holiday season boosts consumer interest")
        print("- X1: Social media advertising (Facebook, TikTok)")
        print("- X2: Search engine advertising")
        print("- Target (T): Sales revenue")
        print("- Competitor Offers (I): Competitor offers")
        print("- Market Growth (G): Market growth")
        print()
        
        # Create causal graph
        dot = Digraph(comment='Business Scenario DAG')
        
        # Add nodes
        dot.node("C", "Christmas", style="dashed")
        dot.node("X1", "Marketing X1")
        dot.node("X2", "Marketing X2")
        dot.node("I", "Competitor Offers", style="dashed")
        dot.node("G", "Market Growth", style="dashed")
        dot.node("T", "Target")
        
        # Add edges
        dot.edge("C", "X1", style="dashed")
        dot.edge("C", "X2", style="dashed")
        dot.edge("I", "X2", style="dashed")
        dot.edge("X1", "X2")
        dot.edge("C", "T", style="dashed")
        dot.edge("X1", "T")
        dot.edge("X2", "T")
        dot.edge("I", "T", style="dashed")
        dot.edge("G", "T", style="dashed")
        
        # Save graph
        try:
            dot.render('business_scenario_dag', format='png', cleanup=True)
            print("Causal graph saved as 'business_scenario_dag.png'")
        except Exception as e:
            print(f"Error saving causal graph: {e}")
            
        return dot
        
    def generate_synthetic_data(self):
        """First try to load real data, generate synthetic data if fails"""
        print("\n5. Data loading process:")
        print("Attempting to load real MMM dataset...")
        
        # First try to load real data
        try:
            return self.load_real_mmm_data()
        except Exception as e:
            print(f"Failed to load real data: {e}")
            print("Falling back to synthetic data generation...")
            return self.generate_synthetic_data_fallback()
    
    def load_real_mmm_data(self):
        """加载真实的MMM数据"""
        try:
            # 读取CSV文件
            data_path = 'data/data_mmm.csv'  # 使用 data_mmm.csv
            self.df = pd.read_csv(data_path)
            
            # 确保日期列是datetime类型
            if 'date_week' in self.df.columns:
                self.df['date_week'] = pd.to_datetime(self.df['date_week'])
                self.date_range = self.df['date_week']
            
            # 准备模型数据
            self.data = self.df.copy()
            if 'date_week' in self.data.columns:
                self.data.rename(columns={'date_week': 'date'}, inplace=True)
            
            print("✅ 成功加载真实数据!")
            print(f"   数据形状: {self.data.shape}")
            print(f"   列: {list(self.data.columns)}")
            if 'y' in self.data.columns:
                print(f"   目标变量统计: 均值={self.data['y'].mean():.2f}, 标准差={self.data['y'].std():.2f}")
            
            return self.df
            
        except Exception as e:
            print(f"❌ 加载真实数据失败: {e}")
            raise
    
    def generate_synthetic_data_fallback(self):
        """Generate synthetic data using the exact user specification"""
        print("Using user-specified data generation method...")
        
        # Create date range
        min_date = pd.to_datetime("2022-01-01")
        max_date = pd.to_datetime("2024-11-06")
        self.date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        
        self.df = pd.DataFrame(data={"date_week": self.date_range}).assign(
            year=lambda x: x["date_week"].dt.year,
            month=lambda x: x["date_week"].dt.month,
            dayofyear=lambda x: x["date_week"].dt.dayofyear,
        )
        
        n = self.df.shape[0]
        print(f"Observation count: {n}")
        
        # Generate market growth
        self.df["market_growth"] = (np.linspace(start=0.0, stop=50, num=n) + 10) ** (1 / 4) - 1
        
        # Generate holiday signal
        holiday_dates = ["24-12"]  # Christmas
        std_devs = [25]  # Holiday influence standard deviation (days)
        holidays_coefficients = [2]  # Holiday influence coefficient
        
        holiday_signal = np.zeros(n)
        for holiday_date, std_dev, coeff in zip(holiday_dates, std_devs, holidays_coefficients):
            for year in self.df['year'].unique():
                holiday_datetime = pd.to_datetime(f"{year}-{holiday_date}")
                if min_date <= holiday_datetime <= max_date:
                    days_diff = (self.date_range - holiday_datetime).days
                    holiday_signal += coeff * np.exp(-0.5 * (days_diff / std_dev) ** 2)
        
        self.df["holiday_signal"] = holiday_signal
        self.df["holiday_contributions"] = holiday_signal * 0.5  # Scale factor
        
        # Generate competitor offers
        rng = np.random.default_rng(self.rng)
        competitor_base = pz.Normal(mu=2, sigma=0.5).rvs(n, random_state=rng)
        competitor_conv = np.convolve(competitor_base, np.ones(7) / 7, mode="same")
        self.df["competitor_offers"] = competitor_conv
        
        # Generate x1 (Social Media)
        x1 = pz.Normal(mu=5, sigma=3).rvs(n, random_state=rng)
        cofounder_effect_holiday_x1 = 2.5
        x1_conv = np.convolve(x1, np.ones(14) / 14, mode="same")
        # Replace first and last 14 values with mean + noise
        noise = pz.Normal(mu=0, sigma=0.1).rvs(28, random_state=rng)
        x1_conv[:14] = x1_conv.mean() + noise[:14]
        x1_conv[-14:] = x1_conv.mean() + noise[14:]
        self.df["x1"] = x1_conv + (holiday_signal * cofounder_effect_holiday_x1)
        
        # Generate x2 (Search Engine)
        x2 = pz.Normal(mu=5, sigma=2).rvs(n, random_state=rng)
        cofounder_effect_holiday_x2 = 2.2
        cofounder_effect_x1_x2 = 1.3
        cofounder_effect_competitor_offers_x2 = -0.7
        x2_conv = np.convolve(x2, np.ones(18) / 12, mode="same")
        # Replace first and last 14 values with mean + noise
        noise = pz.Normal(mu=0, sigma=0.1).rvs(28, random_state=rng)
        x2_conv[:14] = x2_conv.mean() + noise[:14]
        x2_conv[-14:] = x2_conv.mean() + noise[14:]
        self.df["x2"] = (
            x2_conv
            + (holiday_signal * cofounder_effect_holiday_x2)
            + (self.df["x1"] * cofounder_effect_x1_x2)
            + (self.df["competitor_offers"] * cofounder_effect_competitor_offers_x2)
        )
        
        # Apply geometric adstock transformation
        alpha1: float = 0.6
        alpha2: float = 0.2
        
        self.df["x1_adstock"] = (
            geometric_adstock(x=self.df["x1"].to_numpy(), alpha=alpha1, l_max=24, normalize=True)
            .eval()
            .flatten()
        )
        
        self.df["x2_adstock"] = (
            geometric_adstock(x=self.df["x2"].to_numpy(), alpha=alpha2, l_max=24, normalize=True)
            .eval()
            .flatten()
        )
        
        # Apply saturation transformation
        lam1: float = 5.0
        lam2: float = 9.0
        alpha_mm1: float = 6
        alpha_mm2: float = 12
        
        self.df["x1_adstock_saturated"] = michaelis_menten(
            x=self.df["x1_adstock"].to_numpy(), lam=lam1, alpha=alpha_mm1
        )
        
        self.df["x2_adstock_saturated"] = michaelis_menten(
            x=self.df["x2_adstock"].to_numpy(), lam=lam2, alpha=alpha_mm2
        )
        
        # Generate target variable
        self.df["intercept"] = 1.5
        self.df["epsilon"] = rng.normal(loc=0.0, scale=0.08, size=n)
        
        self.df["y"] = (
            self.df["intercept"]
            + self.df["market_growth"]  # implicit coef 1
            - self.df["competitor_offers"]  # explicit coef -1
            + self.df["holiday_contributions"]
            + self.df["x1_adstock_saturated"]
            + self.df["x2_adstock_saturated"]
            + self.df["epsilon"]  # Noise
        )
        
        # Prepare data for modeling
        columns_to_keep = [
            "date_week", 
            "y",
            "x1",
            "x2",
            "holiday_signal"
        ]
        
        self.data = self.df[columns_to_keep].copy()
        self.data.rename(columns={'date_week': 'date', 'y': 'target'}, inplace=True)
        
        print("User-specified synthetic data generation completed!")
        print(f"   Data shape: {self.data.shape}")
        print(f"   Columns: {list(self.data.columns)}")
        print(f"   Target variable stats: Mean={self.data['target'].mean():.2f}, Std={self.data['target'].std():.2f}")
        
        return self.df
    
    def _prepare_model_data(self):
        """准备用于模型训练的数据"""
        try:
            print("🔍 准备模型数据...")
            
            # 确定数据列
            if 'target' in self.df.columns:
                target_col = 'target'
            elif 'y' in self.df.columns:
                target_col = 'y'
            else:
                raise ValueError("数据中没有找到目标变量列 ('target' 或 'y')")
            
            # 确保必要的列存在
            required_cols = ['x1', 'x2']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"数据中缺少必要的列: {missing_cols}")
            
            # 确保date_range存在
            if self.date_range is None:
                if 'date_week' in self.df.columns:
                    self.date_range = pd.to_datetime(self.df['date_week'])
                elif 'date' in self.df.columns:
                    self.date_range = pd.to_datetime(self.df['date'])
                else:
                    # 创建默认的日期范围
                    min_date = pd.to_datetime("2022-01-01")
                    self.date_range = pd.date_range(start=min_date, periods=len(self.df), freq="D")
            
            # 生成缺失的字段
            if 'holiday_signal' not in self.df.columns:
                print("📅 生成假期信号字段...")
                self._generate_holiday_signal_for_real_data()
            
            # 选择用于建模的列
            if 'date_week' in self.df.columns:
                date_col = 'date_week'
            elif 'date' in self.df.columns:
                date_col = 'date'
            else:
                # 如果没有日期列，创建一个索引
                self.df['date_week'] = range(len(self.df))
                date_col = 'date_week'
            
            # 重命名目标变量为'y'以保持一致性
            if target_col != 'y':
                self.df['y'] = self.df[target_col]
            
            # 选择建模所需的列
            columns_to_keep = [date_col, "y", "x1", "x2"]
            
            # 添加控制变量（如果存在）
            control_vars = ["holiday_signal"]
            for var in control_vars:
                if var in self.df.columns:
                    columns_to_keep.append(var)
            
            # 只保留存在的列
            available_columns = [col for col in columns_to_keep if col in self.df.columns]
            
            self.data = self.df[available_columns].copy()
            
            # 确保日期列名为'date'以匹配MMM模型期望
            if date_col != 'date':
                self.data.rename(columns={date_col: 'date'}, inplace=True)
            
            # 确保日期列是datetime类型
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
            
            print(f"✅ 模型数据准备完成")
            print(f"   数据形状: {self.data.shape}")
            print(f"   包含列: {list(self.data.columns)}")
            print(f"   目标变量统计: 均值={self.data['y'].mean():.2f}, 标准差={self.data['y'].std():.2f}")
            
        except Exception as e:
            print(f"❌ 模型数据准备失败: {e}")
            print(f"详细错误: {traceback.format_exc()}")
            # 设置为None，让后续逻辑处理
            self.data = None
            raise
    
    def _generate_holiday_signal_for_real_data(self):
        """为真实数据生成假期信号"""
        try:
            # 假期定义
            holiday_dates = ["24-12"]  # 圣诞节 (MM-DD格式)
            std_devs = [25]  # 假期影响的标准差（天数）
            holidays_coefficients = [2]  # 假期影响系数
            
            # 初始化信号数组
            holiday_signal = np.zeros(len(self.date_range))
            holiday_contributions = np.zeros(len(self.date_range))
            
            print(f"正在为 {len(holiday_dates)} 个假期生成信号...")
            
            # 为每个假期生成信号
            for holiday, std_dev, holiday_coef in zip(holiday_dates, std_devs, holidays_coefficients):
                # 查找假期在日期范围内的所有出现
                holiday_occurrences = self.date_range[self.date_range.dt.strftime("%d-%m") == holiday]
                
                print(f"假期 {holiday} 在数据范围内出现 {len(holiday_occurrences)} 次")
                
                for occurrence in holiday_occurrences:
                    # 计算每个日期与假期的时间差
                    time_diff = (self.date_range - occurrence).days
                    
                    # 使用高斯函数生成假期信号
                    _holiday_signal = np.exp(-0.5 * (time_diff / std_dev) ** 2)
                    
                    # 累加假期信号
                    holiday_signal += _holiday_signal
                    holiday_contributions += _holiday_signal * holiday_coef
            
            # 将生成的信号添加到数据框
            self.df["holiday_signal"] = holiday_signal
            self.df["holiday_contributions"] = holiday_contributions
            
            print(f"✅ 假期信号生成完成")
            print(f"   holiday_signal 范围: [{holiday_signal.min():.4f}, {holiday_signal.max():.4f}]")
            print(f"   holiday_contributions 范围: [{holiday_contributions.min():.4f}, {holiday_contributions.max():.4f}]")
            
        except Exception as e:
            print(f"❌ 假期信号生成失败: {e}")
            # 如果生成失败，创建零值信号
            self.df["holiday_signal"] = np.zeros(len(self.df))
            self.df["holiday_contributions"] = np.zeros(len(self.df))
            print("⚠️ 使用零值假期信号作为回退方案")
    
    def generate_model_evaluation_plots(self, model_result):
        """生成模型拟合图和评估指标"""
        if model_result is None or not hasattr(model_result, 'idata'):
            print("⚠️ 无法生成拟合图：模型结果无效")
            return None
            
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import mean_absolute_percentage_error, r2_score
            import arviz as az
            
            print("\n📊 生成模型评估图表...")
            
            # 获取模型预测
            posterior_predictive = az.extract(model_result.idata, group="posterior_predictive")
            y_pred_scaled = posterior_predictive["y"].mean(dim="sample").values
            y_pred_std_scaled = posterior_predictive["y"].std(dim="sample").values
            
            # 反标准化预测值
            y_pred_mean = model_result.target_transformer.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_pred_std = y_pred_std_scaled * model_result.target_transformer.named_steps['scaler'].scale_.item()
            
            # 实际值
            y_actual = self.df['y'].values[:len(y_pred_mean)]
            
            # 使用正确的R²计算方式
            r2 = az.r2_score(
                y_true=y_actual,
                y_pred=model_result.idata.posterior_predictive.stack(sample=("chain", "draw"))["y"].values.T
                * model_result.target_transformer.named_steps['scaler'].scale_.item(),
            ).iloc[0]
            mape = mean_absolute_percentage_error(y_actual, y_pred_mean)
            mae = np.mean(np.abs(y_actual - y_pred_mean))
            rmse = np.sqrt(np.mean((y_actual - y_pred_mean) ** 2))
            
            print(f"📈 Model evaluation metrics:")
            print(f"   R² Score: {r2:.4f}")
            print(f"   MAPE: {mape:.4f} ({mape*100:.2f}%)")
            print(f"   MAE: {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            
            # Create fit plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Model Fit Quality Assessment', fontsize=16, fontweight='bold')
            
            # 1. Actual vs Predicted values
            axes[0, 0].scatter(y_actual, y_pred_mean, alpha=0.6, color='steelblue')
            min_val = min(y_actual.min(), y_pred_mean.min())
            max_val = max(y_actual.max(), y_pred_mean.max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 时间序列拟合
            if 'date' in self.df.columns:
                dates = self.df['date'].values[:len(y_pred_mean)]
                axes[0, 1].plot(dates, y_actual, label='Actual', color='black', linewidth=2)
                axes[0, 1].plot(dates, y_pred_mean, label='Predicted', color='red', linewidth=1.5)
                axes[0, 1].fill_between(dates, 
                                       y_pred_mean - 1.96*y_pred_std, 
                                       y_pred_mean + 1.96*y_pred_std, 
                                       alpha=0.3, color='red', label='95% Confidence Interval')
                axes[0, 1].set_xlabel('Date')
                axes[0, 1].set_ylabel('Value')
                axes[0, 1].set_title('Time Series Fit')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                # Rotate date labels
                plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
            else:
                axes[0, 1].plot(y_actual, label='Actual', color='black', linewidth=2)
                axes[0, 1].plot(y_pred_mean, label='Predicted', color='red', linewidth=1.5)
                axes[0, 1].fill_between(range(len(y_pred_mean)), 
                                       y_pred_mean - 1.96*y_pred_std, 
                                       y_pred_mean + 1.96*y_pred_std, 
                                       alpha=0.3, color='red', label='95% Confidence Interval')
                axes[0, 1].set_xlabel('Time Point')
                axes[0, 1].set_ylabel('Value')
                axes[0, 1].set_title('Time Series Fit')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Residual plot
            residuals = y_actual - y_pred_mean
            axes[1, 0].scatter(y_pred_mean, residuals, alpha=0.6, color='green')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[1, 0].set_xlabel('Predicted Values')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residual Plot')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Residual distribution
            axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
            axes[1, 1].set_xlabel('Residuals')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title(f'Residual Distribution (RMSE = {rmse:.2f})')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            import os
            os.makedirs('plots', exist_ok=True)
            plot_path = 'plots/model_fit_evaluation.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Model fit plot saved to: {plot_path}")
            
            # Prepare chart data for frontend - user specified charts
            chart_data = {
                'time_series': {
                    'dates': self.df['date'].dt.strftime('%Y-%m-%d').tolist()[:len(y_pred_mean)] if 'date' in self.df.columns else list(range(len(y_pred_mean))),
                    'actual': y_actual.tolist(),
                    'predicted': y_pred_mean.tolist(),
                    'confidence_upper': (y_pred_mean + 1.96*y_pred_std).tolist(),
                    'confidence_lower': (y_pred_mean - 1.96*y_pred_std).tolist()
                }
            }
            
            # Add Media Contribution Recovery data if model has channel_contribution
            if hasattr(model_result, 'fit_result') and hasattr(model_result.fit_result, 'channel_contribution'):
                # Extract channel contributions with confidence intervals
                try:
                    import arviz as az
                    
                    # Get channel contribution data
                    channel_contrib = model_result.fit_result.channel_contribution
                    scale_factor = model_result.target_transformer["scaler"].scale_.item()
                    
                    # Calculate HDI for channel contributions
                    channel_hdi = az.hdi(channel_contrib, hdi_prob=0.95) * scale_factor
                    
                    # Prepare date range
                    dates = self.df['date'].dt.strftime('%Y-%m-%d').tolist() if 'date' in self.df.columns else list(range(len(self.df)))
                    
                    # X1 (Social Media) contribution data
                    x1_mean = (channel_contrib.sel(channel="x1").mean(dim=["chain", "draw"]) * scale_factor).values.tolist()
                    x1_upper = channel_hdi.channel_contribution.isel(hdi=1).sel(channel="x1").values.tolist()
                    x1_lower = channel_hdi.channel_contribution.isel(hdi=0).sel(channel="x1").values.tolist()
                    x1_real = self.df["x1_adstock_saturated"].tolist()
                    
                    # X2 (Search Engine) contribution data  
                    x2_mean = (channel_contrib.sel(channel="x2").mean(dim=["chain", "draw"]) * scale_factor).values.tolist()
                    x2_upper = channel_hdi.channel_contribution.isel(hdi=1).sel(channel="x2").values.tolist()
                    x2_lower = channel_hdi.channel_contribution.isel(hdi=0).sel(channel="x2").values.tolist()
                    x2_real = self.df["x2_adstock_saturated"].tolist()
                    
                    chart_data['media_contribution'] = {
                        'dates': dates,
                        'x1': {
                            'predicted_mean': x1_mean,
                            'confidence_upper': x1_upper,
                            'confidence_lower': x1_lower,
                            'real_effect': x1_real
                        },
                        'x2': {
                            'predicted_mean': x2_mean,
                            'confidence_upper': x2_upper,
                            'confidence_lower': x2_lower,
                            'real_effect': x2_real
                        }
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not extract channel contribution data: {e}")
            
            # Keep actual vs predicted for backward compatibility
            chart_data['actual_vs_predicted'] = {
                'actual': y_actual.tolist(),
                'predicted': y_pred_mean.tolist()
            }
            
            # 返回评估结果
            evaluation_result = {
                'r2_score': r2,
                'mape': mape,
                'mae': mae,
                'rmse': rmse,
                'plot_path': plot_path,
                'sample_size': len(y_actual),
                'prediction_mean': float(y_pred_mean.mean()),
                'prediction_std': float(y_pred_mean.std()),
                'actual_mean': float(y_actual.mean()),
                'actual_std': float(y_actual.std()),
                'chart_data': chart_data  # Add chart data for frontend
            }
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ 生成模型评估图失败: {e}")
            print(f"详细错误: {traceback.format_exc()}")
            return None
        
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
        
    def _extract_control_variables_from_dag(self, dag_string):
        """从DAG字符串中提取控制变量"""
        control_variables = []
        
        # 解析DAG字符串，查找可能的控制变量
        # 控制变量通常是指向x1, x2, y的外生变量
        
        import re
        # 查找所有边关系 (source -> target)
        edge_pattern = r'(\w+)\s*->\s*(\w+)'
        edges = re.findall(edge_pattern, dag_string)
        
        # 收集所有指向处理变量(x1, x2)和结果变量(y)的变量
        sources_to_treatments = set()
        sources_to_outcome = set()
        
        for source, target in edges:
            if target in ['x1', 'x2']:
                sources_to_treatments.add(source)
            elif target in ['y']:
                sources_to_outcome.add(source)
        
        # 控制变量是那些既不是x1, x2, y的变量，但会影响它们的变量
        for source, target in edges:
            if source not in ['x1', 'x2', 'y'] and (target in ['x1', 'x2', 'y']):
                if source not in control_variables:
                    control_variables.append(source)
        
        print(f"🔍 Extracted edges: {edges}")
        print(f"🔍 Sources to treatments: {sources_to_treatments}")
        print(f"🔍 Sources to outcome: {sources_to_outcome}")
        print(f"🔍 Identified control variables: {control_variables}")
        
        return control_variables
    
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
            x1 = rng.normal(loc=5, scale=3, size=n)
            
        cofounder_effect_holiday_x1 = 2.5
        x1_conv = np.convolve(x1, np.ones(14) / 14, mode="same")
        
        # 处理边界效应
        noise = rng.normal(loc=0, scale=0.1, size=28)
        x1_conv[:14] = x1_conv.mean() + noise[:14]
        x1_conv[-14:] = x1_conv.mean() + noise[14:]
        
        self.df["x1"] = x1_conv + (self.df["holiday_signal"] * cofounder_effect_holiday_x1)
        
        # 生成X2 (搜索引擎广告)
        try:
            x2 = np.random.normal(loc=5, scale=2, size=n)
        except:
            x2 = rng.normal(loc=5, scale=2, size=n)
            
        cofounder_effect_holiday_x2 = 2.2
        cofounder_effect_x1_x2 = 1.3
        cofounder_effect_competitor_offers_x2 = -0.7
        
        x2_conv = np.convolve(x2, np.ones(18) / 12, mode="same")
        noise = rng.normal(loc=0, scale=0.1, size=28)
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
        self.df["epsilon"] = rng.normal(loc=0.0, scale=0.08, size=n)
        
        self.df["y"] = (
            self.df["intercept"]
            + self.df["market_growth"]
            - self.df["competitor_offers"]
            + self.df["holiday_contributions"]
            + self.df["x1_adstock_saturated"]
            + self.df["x2_adstock_saturated"]
            + self.df["epsilon"]
        )
        
        # 添加target列以保持兼容性
        self.df["target"] = self.df["y"]
        
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
            
    def run_causal_model(self, version="full", custom_dag=None):
        """Run causal model using user-specified configuration
        
        Args:
            version: Model version ('full', 'simple', etc.)
            custom_dag: User-defined DAG string from UI drag-and-drop. If None, uses default DAG.
        """
        if not PYMC_MARKETING_AVAILABLE:
            print("Skipping causal model - PyMC-Marketing not available")
            return None
            
        print(f"\n7. Causal model ({version}):")
        
        try:
            # Prepare data exactly as user specified
            print(f"🔍 Available data columns: {list(self.data.columns)}")
            X = self.data.drop("y", axis=1)  # Remove target column
            y = self.data["y"]
            print(f"🔍 Feature columns (X): {list(X.columns)}")
            print(f"🔍 Target column shape: {y.shape}")
            
            # Use custom DAG if provided, otherwise use default DAG
            if custom_dag is not None:
                causal_dag = custom_dag
                print("Using user-defined DAG from UI drag-and-drop...")
                print(f"User DAG: {causal_dag}")
            else:
                # Define default causal DAG
                causal_dag = """
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
                print("Using default DAG...")
            
            # Determine available control columns from custom DAG
            available_control_columns = []
            
            if custom_dag is not None:
                # Extract control variables from custom DAG
                control_vars_from_dag = self._extract_control_variables_from_dag(custom_dag)
                print(f"🔍 Control variables extracted from custom DAG: {control_vars_from_dag}")
                
                # Check which control variables are available in data
                for col in control_vars_from_dag:
                    if col in X.columns:
                        available_control_columns.append(col)
            else:
                # Use default control variables
                potential_controls = ["holiday_signal"]
                for col in potential_controls:
                    if col in X.columns:
                        available_control_columns.append(col)
            
            print(f"🔍 Available control columns: {available_control_columns}")
            
            # Create model configuration based on available control columns
            mmm_config = {
                "sampler_config": self.sample_kwargs,
                "date_column": "date",
                "adstock": GeometricAdstock(l_max=24),
                "saturation": MichaelisMentenSaturation(),
                "channel_columns": ["x1", "x2"],
                "outcome_node": "y",
                "time_varying_intercept": True,
            }
            
            # Only add control_columns and dag if we have control variables AND they are not all zero
            if available_control_columns:
                # Check if control variables have non-zero variance
                non_zero_controls = []
                for col in available_control_columns:
                    if col in X.columns and X[col].var() > 1e-10:  # Check if variance is not essentially zero
                        non_zero_controls.append(col)
                
                if non_zero_controls:
                    mmm_config["control_columns"] = non_zero_controls
                    mmm_config["dag"] = causal_dag
                    print(f"🔍 Using causal model with control variables and DAG: {non_zero_controls}")
                else:
                    print("🔍 Control variables have zero variance, using basic MMM model")
                    print("⚠️ Note: Without valid control variables, this will be a standard correlational MMM model")
            else:
                print("🔍 Using basic MMM model without control variables (no DAG constraints)")
                print("⚠️ Note: Without control variables, this will be a standard correlational MMM model")
            
            # Create model with appropriate configuration
            causal_mmm = MMM(**mmm_config)
            
            # Apply user-specified model configuration
            causal_mmm.model_config["intercept_tvp_config"].ls_mu = 180
            causal_mmm.model_config["intercept"] = Prior("Normal", mu=1, sigma=2)
            
            # Display adjustment sets (only if causal model with DAG)
            if hasattr(causal_mmm, 'causal_graphical_model') and causal_mmm.causal_graphical_model is not None:
                print(f"🔍 Adjustment set: {causal_mmm.causal_graphical_model.adjustment_set}")
                print(f"🔍 Minimal adjustment set: {causal_mmm.causal_graphical_model.minimal_adjustment_set}")
            else:
                print("🔍 No causal graphical model (basic MMM mode without DAG constraints)")
            
            try:
                # 使用固定的随机种子进行训练
                causal_mmm.fit(X=X, y=y, target_accept=0.9, random_seed=42)
                causal_mmm.sample_posterior_predictive(
                    X, extend_idata=True, combined=True, random_seed=42
                )
                
                # Check divergences
                divergences = causal_mmm.idata["sample_stats"]["diverging"].sum().item()
                print(f"Causal model divergences: {divergences}")
                
                return causal_mmm
                
            except Exception as e:
                print(f"Model training failed: {str(e)}")
                print("Detailed traceback:")
                import traceback
                print(traceback.format_exc())
                return None
                
        except Exception as e:
            print(f"Error in run_causal_model: {str(e)}")
            print("Detailed traceback:")
            import traceback
            print(traceback.format_exc())
            return None
            
    def compare_models(self, correlational_mmm, causal_mmm):
        """比较不同模型的效果"""
        if not PYMC_MARKETING_AVAILABLE or correlational_mmm is None or causal_mmm is None:
            print("跳过模型比较 - 模型不可用")
            return
        
        # Check if both models have fit_result
        if not hasattr(correlational_mmm, 'fit_result') or correlational_mmm.fit_result is None:
            print("跳过模型比较 - 相关性模型尚未训练完成")
            return
            
        if not hasattr(causal_mmm, 'fit_result') or causal_mmm.fit_result is None:
            print("跳过模型比较 - 因果模型尚未训练完成")
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
        
        # Check if this is a causal model with time-varying intercept
        if not hasattr(causal_mmm, 'causal_graphical_model') or causal_mmm.causal_graphical_model is None:
            print("跳过时变截距绘图 - 此模型没有因果约束或时变截距")
            return
            
        print("\n9. 时变截距分析:")
        print("时变截距可以捕捉未观测到的混淆因子...")
        
        # Check if intercept exists in fit_result
        if not hasattr(causal_mmm, 'fit_result') or causal_mmm.fit_result is None:
            print("模型尚未训练完成，无法分析时变截距")
            return
            
        if "intercept" not in causal_mmm.fit_result:
            print("此模型没有时变截距数据")
            return
        
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
        
    def run_complete_tutorial(self, custom_dag=None):
        """运行完整教程
        
        Args:
            custom_dag: 用户自定义的DAG字符串，如果为None则使用默认DAG
        """
        print("Starting Causal Media Mix Modeling tutorial...")
        
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
        
        # 6. 运行因果模型 - 支持用户自定义DAG
        causal_mmm = self.run_causal_model(version="full", custom_dag=custom_dag)
        
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
    print("Welcome to the Causal Media Mix Modeling tutorial!")
    print("This tutorial will demonstrate how to apply causal inference concepts in MMM.")
    print()
    
    # 创建教程实例
    tutorial = CausalMMMTutorial()
    
    # 运行完整教程
    results = tutorial.run_complete_tutorial()
    
    return results

if __name__ == "__main__":
    results = main() 