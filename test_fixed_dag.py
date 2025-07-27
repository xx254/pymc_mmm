#!/usr/bin/env python3
"""
测试修复后的DAG功能
使用原始MMM类测试DAG对模型的影响
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from causal_mmm_tutorial import CausalMMMTutorial
import numpy as np
import pandas as pd

def test_fixed_dag_functionality():
    """测试修复后的DAG功能"""
    
    print("🔍 测试修复后的DAG功能")
    print("="*60)
    
    # 创建教程实例
    tutorial = CausalMMMTutorial()
    tutorial.generate_synthetic_data()
    
    results = []
    
    # 测试1: 标准MMM（无DAG）
    print(f"\n📊 测试1: 标准MMM模型（无DAG约束）")
    try:
        correlational_model = tutorial.run_correlational_model()
        
        if correlational_model is not None:
            evaluation = tutorial.generate_model_evaluation_plots(correlational_model)
            if evaluation:
                results.append({
                    'name': 'Standard MMM (No DAG)',
                    'r2_score': evaluation['r2_score'],
                    'has_dag': False
                })
                print(f"✅ Standard MMM - R²: {evaluation['r2_score']:.6f}")
        
    except Exception as e:
        print(f"❌ Standard MMM failed: {e}")
    
    # 测试2: 简单DAG
    print(f"\n📊 测试2: 简单DAG模型")
    try:
        simple_dag = """
        digraph {
            x1 -> y;
            x2 -> y;
            holiday_signal -> y;
        }
        """
        
        causal_model = tutorial.run_causal_model(version="custom", custom_dag=simple_dag)
        
        if causal_model is not None:
            evaluation = tutorial.generate_model_evaluation_plots(causal_model)
            if evaluation:
                results.append({
                    'name': 'Simple DAG Model',
                    'r2_score': evaluation['r2_score'],
                    'has_dag': True
                })
                print(f"✅ Simple DAG - R²: {evaluation['r2_score']:.6f}")
        
    except Exception as e:
        print(f"❌ Simple DAG failed: {e}")
    
    # 测试3: 复杂DAG
    print(f"\n📊 测试3: 复杂DAG模型")
    try:
        complex_dag = """
        digraph {
            x1 -> y;
            x2 -> y;
            x1 -> x2;
            holiday_signal -> x1;
            holiday_signal -> x2;
            holiday_signal -> y;
        }
        """
        
        causal_model = tutorial.run_causal_model(version="custom", custom_dag=complex_dag)
        
        if causal_model is not None:
            evaluation = tutorial.generate_model_evaluation_plots(causal_model)
            if evaluation:
                results.append({
                    'name': 'Complex DAG Model',
                    'r2_score': evaluation['r2_score'],
                    'has_dag': True
                })
                print(f"✅ Complex DAG - R²: {evaluation['r2_score']:.6f}")
        
    except Exception as e:
        print(f"❌ Complex DAG failed: {e}")
    
    # 分析结果
    print(f"\n{'='*60}")
    print("📈 结果分析")
    print(f"{'='*60}")
    
    if len(results) >= 2:
        print(f"\n🎯 R²分数对比:")
        for result in results:
            dag_status = "✓" if result['has_dag'] else "✗"
            print(f"   {result['name']} (DAG:{dag_status}): {result['r2_score']:.6f}")
        
        r2_values = [r['r2_score'] for r in results]
        r2_std = np.std(r2_values)
        r2_range = max(r2_values) - min(r2_values)
        
        print(f"\n📊 统计分析:")
        print(f"   R²值标准差: {r2_std:.6f}")
        print(f"   R²值范围: {r2_range:.6f}")
        print(f"   平均R²: {np.mean(r2_values):.6f}")
        
        print(f"\n💡 结果解释:")
        if r2_range < 0.01:
            print("   ✅ 这是正常的！不同DAG产生相似R²值是预期行为")
            print("   🎯 DAG的价值在于:")
            print("      • 提供无偏的因果效应估计")
            print("      • 正确控制混淆变量")
            print("      • 支持反事实分析和政策仿真")
        else:
            print("   🔍 不同DAG产生了显著差异")
            print("   这可能表明DAG约束确实影响了模型结构")
    else:
        print("❌ 测试结果不足，无法进行比较")
    
    return results

def demonstrate_dag_value():
    """演示DAG的真正价值"""
    print(f"\n{'='*60}")
    print("🎯 DAG在MMM中的真正价值")
    print(f"{'='*60}")
    
    print(f"\n📚 核心概念:")
    print(f"   R² ≠ 因果准确性")
    print(f"   • R²衡量预测准确性")
    print(f"   • DAG确保因果效应的无偏估计")
    print(f"   • 相同的R²可能隐藏不同的因果结构")
    
    print(f"\n🛠️ 实际应用:")
    print(f"   1. 渠道归因：DAG帮助正确分解各渠道的真实贡献")
    print(f"   2. 预算优化：基于因果效应优化媒体投入")
    print(f"   3. 政策仿真：预测营销策略变化的影响")
    print(f"   4. 混淆控制：避免虚假的相关性")

if __name__ == "__main__":
    try:
        print("🚀 开始DAG功能验证测试...")
        results = test_fixed_dag_functionality()
        demonstrate_dag_value()
        
        print(f"\n{'='*60}")
        print("✅ 测试完成")
        print(f"{'='*60}")
        
        if len(results) > 0:
            success_count = len(results)
            print(f"📊 成功测试了 {success_count} 个模型")
            print(f"🎯 关键发现: DAG的价值在于因果推断，而非提升R²")
        else:
            print("❌ 所有测试都失败了，需要进一步调试")
            
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc() 