#!/usr/bin/env python3
"""
控制实验：测试纯粹DAG结构差异对R²的影响
保持所有其他配置相同，只改变DAG结构
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from causal_mmm_tutorial import CausalMMMTutorial
import numpy as np
import pandas as pd

def test_pure_dag_effect():
    """测试纯粹DAG结构差异的影响"""
    
    print("🔍 控制实验：纯粹DAG结构差异对R²的影响")
    print("="*70)
    print("🎯 实验设计:")
    print("   • 所有配置完全相同 (Time Varying Intercept=True)")
    print("   • 控制变量相同 (holiday_signal)")
    print("   • 采样参数相同")
    print("   • 只改变DAG结构")
    print("="*70)
    
    # 创建教程实例
    tutorial = CausalMMMTutorial()
    tutorial.generate_synthetic_data()
    
    # 定义测试用的DAG结构 - 保持配置相同，只改变DAG
    test_configs = [
        {
            'name': 'Simple DAG (直接因果)',
            'dag': """
            digraph {
                x1 -> y;
                x2 -> y;
                holiday_signal -> y;
            }
            """,
            'description': '最简单的直接因果关系'
        },
        {
            'name': 'Confounded DAG (有混淆)',
            'dag': """
            digraph {
                x1 -> y;
                x2 -> y;
                holiday_signal -> y;
                holiday_signal -> x1;
                holiday_signal -> x2;
            }
            """,
            'description': 'holiday_signal作为混淆变量'
        },
        {
            'name': 'Interactive DAG (有交互)',
            'dag': """
            digraph {
                x1 -> y;
                x2 -> y;
                x1 -> x2;
                holiday_signal -> y;
                holiday_signal -> x1;
                holiday_signal -> x2;
            }
            """,
            'description': 'x1和x2之间有因果关系'
        }
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n📊 测试 {i+1}: {config['name']}")
        print(f"   描述: {config['description']}")
        print(f"   DAG:\n{config['dag']}")
        
        try:
            # 使用完全相同的配置运行模型
            model = run_controlled_model(tutorial, config['dag'])
            
            if model is not None:
                # 获取R²评估
                evaluation = tutorial.generate_model_evaluation_plots(model)
                
                if evaluation:
                    result_info = {
                        'name': config['name'],
                        'r2_score': evaluation['r2_score'],
                        'mape': evaluation['mape'],
                        'dag_structure': config['dag'].strip(),
                        'description': config['description']
                    }
                    
                    results.append(result_info)
                    
                    print(f"✅ 成功! R²: {evaluation['r2_score']:.6f}, MAPE: {evaluation['mape']:.4f}")
                    
                    # 显示DAG分析结果
                    if hasattr(model, 'causal_graphical_model') and model.causal_graphical_model is not None:
                        try:
                            adj_set = model.causal_graphical_model.adjustment_set
                            min_adj_set = model.causal_graphical_model.minimal_adjustment_set
                            print(f"   📋 Adjustment set: {adj_set}")
                            print(f"   📋 Minimal adjustment set: {min_adj_set}")
                            print(f"   📋 Final control columns: {model.control_columns}")
                        except:
                            print(f"   ⚠️ 无法获取adjustment set信息")
                else:
                    print(f"❌ 评估失败")
            else:
                print(f"❌ 模型训练失败")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 分析结果
    analyze_pure_dag_results(results)
    
    return results

def run_controlled_model(tutorial, dag_string):
    """运行控制实验模型 - 所有配置相同，只改变DAG"""
    try:
        from pymc_marketing.mmm import MMM, GeometricAdstock, MichaelisMentenSaturation
        from pymc_marketing.prior import Prior
        
        # 准备数据
        X = tutorial.data.drop("y", axis=1)
        y = tutorial.data["y"]
        
        print(f"🔧 创建模型 (统一配置)...")
        
        # 方法1: 不使用DAG的自动control_columns决策，强制使用相同的控制变量
        # 这样确保所有DAG都使用完全相同的control_columns
        model = MMM(
            sampler_config=tutorial.sample_kwargs,
            date_column="date",
            adstock=GeometricAdstock(l_max=24),
            saturation=MichaelisMentenSaturation(),
            channel_columns=["x1", "x2"],
            # 不传递control_columns，让DAG分析自动决定
            # 这样避免验证错误，同时保持配置一致性
            outcome_node="y",
            dag=dag_string,  # 唯一的差异
            time_varying_intercept=True,  # 统一使用True
        )
        
        # 手动确保控制变量一致 - 在模型创建后强制设置
        if hasattr(model, 'control_columns'):
            # 无论DAG分析结果如何，都强制使用holiday_signal作为控制变量
            # 但只有当数据中存在且有方差时才设置
            if "holiday_signal" in X.columns and X["holiday_signal"].var() > 1e-10:
                print(f"🔧 强制设置控制变量: ['holiday_signal']")
                model.control_columns = ["holiday_signal"]
            else:
                print(f"🔧 控制变量设置为None（数据中无holiday_signal或方差为0）")
                model.control_columns = None
        
        # 统一的模型配置
        model.model_config["intercept_tvp_config"].ls_mu = 180
        model.model_config["intercept"] = Prior("Normal", mu=1, sigma=2)
        
        print(f"🏃 训练模型...")
        print(f"   📋 最终控制变量: {model.control_columns}")
        
        # 显示DAG分析结果（仅用于信息，不影响训练）
        if hasattr(model, 'causal_graphical_model') and model.causal_graphical_model is not None:
            try:
                adj_set = model.causal_graphical_model.adjustment_set
                min_adj_set = model.causal_graphical_model.minimal_adjustment_set
                print(f"   📋 DAG建议的adjustment set: {adj_set}")
                print(f"   📋 DAG建议的minimal adjustment set: {min_adj_set}")
                print(f"   💡 注意：我们忽略DAG建议，强制使用相同的控制变量以保证实验公平性")
            except:
                print(f"   ⚠️ 无法获取DAG分析结果")
        
        # 统一的训练参数
        model.fit(X=X, y=y, target_accept=0.95, random_seed=42)
        model.sample_posterior_predictive(X, extend_idata=True, combined=True, random_seed=42)
        
        # 检查收敛性
        divergences = model.idata["sample_stats"]["diverging"].sum().item()
        print(f"   🔍 Divergences: {divergences}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型创建/训练失败: {e}")
        print(f"详细错误信息:")
        import traceback
        traceback.print_exc()
        return None

def analyze_pure_dag_results(results):
    """分析纯粹DAG结构差异的结果"""
    print(f"\n{'='*70}")
    print("📈 控制实验结果分析")
    print(f"{'='*70}")
    
    if len(results) < 2:
        print("❌ 结果不足，无法分析")
        return
    
    print(f"\n🎯 R²分数对比:")
    for result in results:
        print(f"   {result['name']}: {result['r2_score']:.6f}")
    
    r2_values = [r['r2_score'] for r in results]
    r2_std = np.std(r2_values)
    r2_range = max(r2_values) - min(r2_values)
    r2_mean = np.mean(r2_values)
    
    print(f"\n📊 统计分析:")
    print(f"   R²平均值: {r2_mean:.6f}")
    print(f"   R²标准差: {r2_std:.6f}")
    print(f"   R²值范围: {r2_range:.6f}")
    print(f"   变异系数: {(r2_std/r2_mean)*100:.2f}%")
    
    print(f"\n💡 结果解释:")
    
    if r2_range > 0.01:  # 1%以上的差异认为是有意义的
        print(f"   ✅ 有意义的差异!")
        print(f"   📊 R²范围: {r2_range:.4f} (> 0.01)")
        print(f"   🎯 结论: 即使在相同配置下，不同DAG结构仍然影响模型性能")
        
        # 找出表现最好和最差的
        best_result = max(results, key=lambda x: x['r2_score'])
        worst_result = min(results, key=lambda x: x['r2_score'])
        
        print(f"\n🏆 最佳DAG: {best_result['name']} (R² = {best_result['r2_score']:.6f})")
        print(f"   特点: {best_result['description']}")
        print(f"📉 最差DAG: {worst_result['name']} (R² = {worst_result['r2_score']:.6f})")
        print(f"   特点: {worst_result['description']}")
        
        improvement = ((best_result['r2_score'] - worst_result['r2_score']) / worst_result['r2_score']) * 100
        print(f"🔢 性能提升: {improvement:.2f}%")
        
    elif r2_range > 0.001:  # 0.1%以上的差异
        print(f"   🤔 轻微差异")
        print(f"   📊 R²范围: {r2_range:.4f} (0.001-0.01)")
        print(f"   🎯 结论: DAG结构有轻微影响，但不如其他配置因素重要")
        
    else:  # 差异很小
        print(f"   😐 差异微小")
        print(f"   📊 R²范围: {r2_range:.4f} (< 0.001)")
        print(f"   🎯 结论: 在相同配置下，纯粹的DAG结构差异对R²影响很小")
    
    print(f"\n🔬 理论分析:")
    print(f"   • DAG主要影响因果效应估计的准确性，而不是预测性能")
    print(f"   • Time Varying Intercept等配置对R²的影响可能更大")
    print(f"   • 不同DAG的adjustment set可能影响哪些变量被控制")
    print(f"   • 相同的数据和特征限制了模型性能的差异空间")
    
    print(f"\n🎯 实践启示:")
    if r2_range > 0.01:
        print(f"   • DAG结构选择确实重要，不仅影响因果解释也影响预测性能")
        print(f"   • 应该基于领域知识选择最合适的DAG结构")
    else:
        print(f"   • DAG的主要价值在于因果推断准确性")
        print(f"   • 选择DAG时应更关注因果关系的正确性而非R²值")

if __name__ == "__main__":
    try:
        print("🚀 开始纯粹DAG结构差异测试...")
        
        results = test_pure_dag_effect()
        
        print(f"\n{'='*70}")
        print("✅ 控制实验完成")
        print(f"{'='*70}")
        
        if len(results) >= 2:
            r2_range = max(r['r2_score'] for r in results) - min(r['r2_score'] for r in results)
            if r2_range > 0.01:
                print(f"🎯 重要发现：纯粹DAG差异确实影响R²值!")
                print(f"📊 差异程度: {r2_range:.4f}")
            else:
                print(f"🎯 发现：纯粹DAG差异对R²影响较小")
                print(f"📊 差异程度: {r2_range:.4f}")
                print(f"💡 DAG的主要价值在于因果推断准确性")
        else:
            print("❌ 实验失败：无法获得足够的比较数据")
            
    except Exception as e:
        print(f"❌ 实验过程出错: {e}")
        import traceback
        traceback.print_exc() 