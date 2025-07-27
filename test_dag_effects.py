#!/usr/bin/env python3
"""
测试修复后的DAG集成效果
分析DAG对MMM模型的真正影响
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from causal_mmm_tutorial import CausalMMMTutorial
import numpy as np
import pandas as pd

def test_dag_integration():
    """测试DAG是否正确集成到模型中"""
    
    print("🔍 测试修复后的DAG集成效果")
    print("="*60)
    
    # 创建教程实例
    tutorial = CausalMMMTutorial()
    tutorial.generate_synthetic_data()
    
    # 定义对比测试
    test_cases = [
        {
            'name': 'Standard MMM (No DAG)',
            'method': 'correlational',
            'description': '标准MMM模型，无DAG约束'
        },
        {
            'name': 'Simple Causal DAG',
            'method': 'causal',
            'dag_string': '''
            digraph {
                x1 -> y;
                x2 -> y;
                holiday_signal -> y;
            }
            ''',
            'description': '简单因果DAG，包含基本因果关系'
        },
        {
            'name': 'Complex Causal DAG',
            'method': 'causal', 
            'dag_string': '''
            digraph {
                x1 -> y;
                x2 -> y;
                x1 -> x2;
                holiday_signal -> x1;
                holiday_signal -> x2;
                holiday_signal -> y;
            }
            ''',
            'description': '复杂因果DAG，包含混淆变量和交互'
        },
        {
            'name': 'Minimal DAG',
            'method': 'causal',
            'dag_string': '''
            digraph {
                x1 -> y;
                x2 -> y;
            }
            ''',
            'description': '最小DAG，仅包含直接因果关系'
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n📊 测试案例 {i+1}: {case['name']}")
        print(f"描述: {case['description']}")
        print("-" * 50)
        
        try:
            # 根据方法类型运行不同的模型
            if case['method'] == 'correlational':
                print("🔍 运行标准MMM模型...")
                model_result = tutorial.run_correlational_model()
            else:
                print("🔍 运行带DAG约束的因果模型...")
                model_result = tutorial.run_causal_model(
                    version="custom", 
                    custom_dag=case['dag_string']
                )
            
            if model_result is not None:
                # 分析模型结构
                causal_analysis = analyze_causal_constraints(model_result)
                
                # 获取评估结果
                evaluation = tutorial.generate_model_evaluation_plots(model_result)
                
                if evaluation:
                    result_info = {
                        'name': case['name'],
                        'method': case['method'],
                        'r2_score': evaluation['r2_score'],
                        'mape': evaluation['mape'],
                        'mae': evaluation['mae'],
                        'rmse': evaluation['rmse'],
                        'causal_analysis': causal_analysis,
                        'model_has_dag': causal_analysis['has_causal_model']
                    }
                    
                    results.append(result_info)
                    
                    print(f"✅ 训练成功!")
                    print(f"   R² Score: {evaluation['r2_score']:.6f}")
                    print(f"   MAPE: {evaluation['mape']:.6f} ({evaluation['mape']*100:.2f}%)")
                    print(f"   MAE: {evaluation['mae']:.2f}")
                    print(f"   RMSE: {evaluation['rmse']:.2f}")
                    print(f"   有因果约束: {causal_analysis['has_causal_model']}")
                    if causal_analysis['has_causal_model']:
                        print(f"   调整集合: {causal_analysis['adjustment_set']}")
                        print(f"   控制变量: {causal_analysis['control_columns']}")
                else:
                    print("❌ 评估失败")
            else:
                print("❌ 训练失败")
                
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 分析结果
    analyze_results(results)
    
    return results

def analyze_causal_constraints(model_result):
    """分析模型的因果约束"""
    analysis = {
        'has_causal_model': False,
        'adjustment_set': None,
        'control_columns': None,
        'dag_structure': None
    }
    
    try:
        # 检查是否有因果图模型
        if hasattr(model_result, 'causal_graphical_model') and model_result.causal_graphical_model is not None:
            analysis['has_causal_model'] = True
            analysis['adjustment_set'] = str(model_result.causal_graphical_model.adjustment_set)
            if hasattr(model_result.causal_graphical_model, 'minimal_adjustment_set'):
                analysis['minimal_adjustment_set'] = str(model_result.causal_graphical_model.minimal_adjustment_set)
        
        # 检查控制变量
        if hasattr(model_result, 'control_columns'):
            analysis['control_columns'] = model_result.control_columns
        
        # 检查DAG结构
        if hasattr(model_result, 'dag'):
            analysis['dag_structure'] = model_result.dag
            
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis

def analyze_results(results):
    """分析测试结果"""
    print(f"\n{'='*60}")
    print("📈 测试结果分析")
    print(f"{'='*60}")
    
    if len(results) == 0:
        print("❌ 没有成功的测试结果")
        return
    
    # 分离有DAG约束和无DAG约束的模型
    causal_models = [r for r in results if r['model_has_dag']]
    standard_models = [r for r in results if not r['model_has_dag']]
    
    print(f"\n🔍 模型分类:")
    print(f"   标准MMM模型 (无DAG约束): {len(standard_models)} 个")
    print(f"   因果MMM模型 (有DAG约束): {len(causal_models)} 个")
    
    # R²值对比
    print(f"\n🎯 R²分数对比:")
    for result in results:
        dag_status = "✓DAG" if result['model_has_dag'] else "✗DAG"
        print(f"   {result['name']} ({dag_status}): {result['r2_score']:.6f}")
    
    if len(results) >= 2:
        r2_values = [r['r2_score'] for r in results]
        r2_std = np.std(r2_values)
        r2_range = max(r2_values) - min(r2_values)
        
        print(f"\n📊 统计分析:")
        print(f"   R²值标准差: {r2_std:.6f}")
        print(f"   R²值范围: {r2_range:.6f}")
        print(f"   平均R²: {np.mean(r2_values):.6f}")
        
        # 解释结果
        print(f"\n💡 结果解释:")
        if r2_range < 0.01:
            print("   ✅ 不同模型的R²值相近是正常的，因为:")
            print("      • R²衡量的是预测准确性，不是因果解释能力")
            print("      • 所有模型使用相同的特征数据进行预测")
            print("      • DAG的价值在于因果推断，而非预测性能提升")
            print()
            print("   🎯 DAG的真正价值体现在:")
            print("      • 提供无偏的因果效应估计")
            print("      • 正确处理混淆变量")
            print("      • 支持反事实分析和政策仿真")
            print("      • 渠道贡献的因果分解")
        else:
            print("   🔍 不同模型产生了显著不同的R²值")
            print("      这可能表明DAG约束影响了模型的拟合能力")
    
    # 因果约束分析
    print(f"\n🔬 因果约束详细分析:")
    for result in results:
        print(f"\n--- {result['name']} ---")
        analysis = result['causal_analysis']
        print(f"   有因果约束: {analysis.get('has_causal_model', False)}")
        if analysis.get('has_causal_model'):
            print(f"   调整集合: {analysis.get('adjustment_set', 'N/A')}")
            print(f"   控制变量: {analysis.get('control_columns', 'N/A')}")
        else:
            print(f"   模型类型: 标准MMM（无因果约束）")

def demonstrate_causal_benefits():
    """演示DAG在因果推断中的价值"""
    print(f"\n{'='*60}")
    print("🎯 DAG在因果推断中的价值演示")
    print(f"{'='*60}")
    
    print("\n1. 📚 理论价值:")
    print("   • 混淆控制: DAG帮助识别需要控制的变量")
    print("   • 因果识别: 确保估计的是真实因果效应，而非虚假关联")
    print("   • 透明性: 明确展示建模假设和因果假设")
    
    print("\n2. 🛠️ 实践价值:")
    print("   • 政策仿真: 预测改变媒体投入的因果效果")
    print("   • 预算优化: 基于因果效应优化媒体预算分配")
    print("   • 归因分析: 提供更准确的渠道贡献归因")
    
    print("\n3. ⚠️ 常见误解:")
    print("   • DAG不是为了提高R²，而是为了因果推断")
    print("   • 相同的R²可能隐藏不同的因果结构")
    print("   • 预测准确性 ≠ 因果解释准确性")

if __name__ == "__main__":
    try:
        print("🚀 开始DAG集成效果测试...")
        results = test_dag_integration()
        demonstrate_causal_benefits()
        
        print(f"\n{'='*60}")
        print("✅ 测试完成")
        print(f"{'='*60}")
        
        print("\n📝 总结建议:")
        print("1. 不同DAG产生相似R²值是正常现象")
        print("2. 关注因果效应估计的准确性，而非预测准确性")
        print("3. 使用DAG进行反事实分析和政策仿真")
        print("4. 验证混淆变量是否被正确控制")
        
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc() 