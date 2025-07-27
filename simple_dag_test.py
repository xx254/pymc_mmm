#!/usr/bin/env python3
"""
简单的DAG功能测试
直接使用原始MMM类测试DAG功能是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from causal_mmm_tutorial import CausalMMMTutorial
import numpy as np
import pandas as pd

# 直接从pymc_marketing导入原始MMM，而不是使用ProgressMMM
try:
    from pymc_marketing.mmm import MMM as OriginalMMM
    from pymc_marketing.mmm import GeometricAdstock, MichaelisMentenSaturation
    from pymc_marketing.prior import Prior
    ORIGINAL_MMM_AVAILABLE = True
except ImportError as e:
    print(f"无法导入原始MMM: {e}")
    ORIGINAL_MMM_AVAILABLE = False

def test_simple_dag_functionality():
    """测试简单的DAG功能"""
    
    print("🔍 测试基础DAG功能")
    print("="*50)
    
    if not ORIGINAL_MMM_AVAILABLE:
        print("❌ 原始MMM不可用，跳过测试")
        return
    
    # 创建教程实例并生成数据
    tutorial = CausalMMMTutorial()
    tutorial.generate_synthetic_data()
    
    # 准备数据
    X = tutorial.data.drop("y", axis=1)
    y = tutorial.data["y"]
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"特征列: {list(X.columns)}")
    
    # 测试1: 标准MMM（无DAG）
    print(f"\n📊 测试1: 标准MMM模型（无DAG约束）")
    try:
        standard_config = {
            "date_column": "date",
            "channel_columns": ["x1", "x2"],
            "adstock": GeometricAdstock(l_max=24),
            "saturation": MichaelisMentenSaturation(),
            "control_columns": ["holiday_signal"] if "holiday_signal" in X.columns else None
        }
        
        standard_mmm = OriginalMMM(**standard_config)
        print(f"✅ 标准MMM创建成功")
        print(f"   控制变量: {standard_mmm.control_columns}")
        print(f"   有因果图: {hasattr(standard_mmm, 'causal_graphical_model') and standard_mmm.causal_graphical_model is not None}")
        
    except Exception as e:
        print(f"❌ 标准MMM创建失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 带DAG的因果MMM
    print(f"\n📊 测试2: 带DAG约束的因果MMM模型")
    try:
        dag_string = """
        digraph {
            x1 -> y;
            x2 -> y;
            holiday_signal -> y;
        }
        """
        
        causal_config = {
            "date_column": "date", 
            "channel_columns": ["x1", "x2"],
            "adstock": GeometricAdstock(l_max=24),
            "saturation": MichaelisMentenSaturation(),
            "dag": dag_string,
            "treatment_nodes": ["x1", "x2"],
            "outcome_node": "y"
        }
        
        causal_mmm = OriginalMMM(**causal_config)
        print(f"✅ 因果MMM创建成功")
        print(f"   控制变量: {causal_mmm.control_columns}")
        print(f"   有因果图: {hasattr(causal_mmm, 'causal_graphical_model') and causal_mmm.causal_graphical_model is not None}")
        
        if hasattr(causal_mmm, 'causal_graphical_model') and causal_mmm.causal_graphical_model is not None:
            cgm = causal_mmm.causal_graphical_model
            
            # 检查adjustment_set属性
            if hasattr(cgm, 'adjustment_set'):
                print(f"   调整集合: {cgm.adjustment_set}")
            else:
                print(f"   ⚠️ 调整集合属性不存在")
                
            # 检查minimal_adjustment_set属性
            if hasattr(cgm, 'minimal_adjustment_set'):
                print(f"   最小调整集合: {cgm.minimal_adjustment_set}")
            else:
                print(f"   ⚠️ 最小调整集合属性不存在")
        
    except Exception as e:
        print(f"❌ 因果MMM创建失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试3: 实际训练一个简单模型
    print(f"\n📊 测试3: 训练一个简单模型")
    try:
        # 使用最小配置
        minimal_config = {
            "date_column": "date",
            "channel_columns": ["x1", "x2"],
            "adstock": GeometricAdstock(l_max=8),  # 减小参数
            "saturation": MichaelisMentenSaturation(),
            "sampler_config": {
                "draws": 100,  # 很少的采样次数用于快速测试
                "chains": 2,
                "tune": 100
            }
        }
        
        test_mmm = OriginalMMM(**minimal_config)
        print(f"✅ 测试模型创建成功")
        
        # 尝试fit（但不等待完成，只是检查是否能开始）
        print(f"🔍 开始模型训练（快速测试）...")
        result = test_mmm.fit(X=X, y=y, target_accept=0.8, random_seed=42)
        print(f"✅ 模型训练完成！")
        
        # 生成评估
        evaluation = tutorial.generate_model_evaluation_plots(test_mmm)
        if evaluation:
            print(f"📈 模型评估:")
            print(f"   R² Score: {evaluation['r2_score']:.6f}")
            print(f"   MAPE: {evaluation['mape']:.6f}")
            return evaluation
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    try:
        print("🚀 开始简单DAG功能测试...")
        result = test_simple_dag_functionality()
        
        if result:
            print(f"\n✅ 测试成功完成!")
            print(f"模型R²: {result['r2_score']:.6f}")
        else:
            print(f"\n⚠️ 测试完成但没有生成评估结果")
            
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc() 