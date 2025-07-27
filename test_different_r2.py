#!/usr/bin/env python3
"""
测试不同DAG是否产生不同的R²值
模仿notebook中的两个例子
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from causal_mmm_tutorial import CausalMMMTutorial
import numpy as np
import pandas as pd

def test_different_r2_values():
    """测试不同DAG是否产生不同的R²值"""
    
    print("🔍 测试不同DAG产生不同R²值")
    print("="*60)
    print("📚 基于notebook例子:")
    print("   例子1 (简单DAG): R² ≈ 0.84")
    print("   例子2 (复杂DAG): R² ≈ 0.99")
    print()
    
    # 创建教程实例
    tutorial = CausalMMMTutorial()
    tutorial.generate_synthetic_data()
    
    results = []
    
    # 测试1: 简单DAG (期望R² ≈ 0.84)
    print(f"\n📊 测试1: 简单DAG模型 (模仿notebook例子1)")
    print("   特点: time_varying_intercept=False, 简单DAG结构")
    try:
        simple_model = tutorial.run_causal_model(version="simple")
        
        if simple_model is not None:
            evaluation = tutorial.generate_model_evaluation_plots(simple_model)
            if evaluation:
                results.append({
                    'name': 'Simple DAG (notebook例子1风格)',
                    'r2_score': evaluation['r2_score'],
                    'model_type': 'simple',
                    'time_varying_intercept': False
                })
                print(f"✅ 简单DAG - R²: {evaluation['r2_score']:.6f}")
            else:
                print("❌ 评估失败")
        else:
            print("❌ 模型训练失败")
        
    except Exception as e:
        print(f"❌ 简单DAG测试失败: {e}")
    
    # 测试2: 复杂DAG (期望R² ≈ 0.99)
    print(f"\n📊 测试2: 复杂DAG模型 (模仿notebook例子2)")
    print("   特点: time_varying_intercept=True, 复杂DAG结构")
    try:
        complex_model = tutorial.run_causal_model(version="full")
        
        if complex_model is not None:
            evaluation = tutorial.generate_model_evaluation_plots(complex_model)
            if evaluation:
                results.append({
                    'name': 'Complex DAG (notebook例子2风格)',
                    'r2_score': evaluation['r2_score'],
                    'model_type': 'full',
                    'time_varying_intercept': True
                })
                print(f"✅ 复杂DAG - R²: {evaluation['r2_score']:.6f}")
            else:
                print("❌ 评估失败")
        else:
            print("❌ 模型训练失败")
        
    except Exception as e:
        print(f"❌ 复杂DAG测试失败: {e}")
    
    # 分析结果
    print(f"\n{'='*60}")
    print("📈 结果对比分析")
    print(f"{'='*60}")
    
    if len(results) >= 2:
        print(f"\n🎯 R²分数对比:")
        for result in results:
            tvp_status = "✓TVP" if result['time_varying_intercept'] else "✗TVP"
            print(f"   {result['name']} ({tvp_status}): {result['r2_score']:.6f}")
        
        r2_values = [r['r2_score'] for r in results]
        r2_std = np.std(r2_values)
        r2_range = max(r2_values) - min(r2_values)
        
        print(f"\n📊 统计分析:")
        print(f"   R²值标准差: {r2_std:.6f}")
        print(f"   R²值范围: {r2_range:.6f}")
        print(f"   平均R²: {np.mean(r2_values):.6f}")
        
        print(f"\n💡 结果解释:")
        if r2_range > 0.05:  # 期望有显著差异
            print("   ✅ 成功！不同DAG产生了显著不同的R²值")
            print("   🎯 这证明了DAG结构确实影响模型性能")
            print(f"   📊 差异: {r2_range:.4f} (期望 > 0.05)")
            
            # 找出最佳模型
            best_result = max(results, key=lambda x: x['r2_score'])
            print(f"   🏆 最佳模型: {best_result['name']} (R² = {best_result['r2_score']:.6f})")
            
        elif r2_range > 0.01:
            print("   🤔 有一定差异，但不如预期显著")
            print(f"   📊 差异: {r2_range:.4f} (期望 > 0.05)")
            print("   💭 可能原因: 数据特性、模型配置等")
            
        else:
            print("   ❌ 仍然没有显著差异")
            print(f"   📊 差异: {r2_range:.4f} (太小)")
            print("   🔍 需要进一步调试模型配置")
            
        # 分析原因
        print(f"\n🔬 技术分析:")
        print(f"   关键差异因素:")
        print(f"   • Time Varying Intercept: 复杂模型有，简单模型没有")
        print(f"   • DAG结构: 复杂模型有更多因果关系")
        print(f"   • 控制变量: 两者都使用holiday_signal")
        print(f"   • Target Accept: 复杂模型0.95，简单模型0.90")
        
    else:
        print("❌ 测试结果不足，无法进行比较")
    
    return results

def explain_notebook_differences():
    """解释notebook中两个例子的差异"""
    print(f"\n{'='*60}")
    print("📚 Notebook例子分析")
    print(f"{'='*60}")
    
    print(f"\n🔍 例子1 (R² = 0.84) 特点:")
    print(f"   • 简单DAG: x1->y, x2->y, holiday_signal->y等")
    print(f"   • 没有time_varying_intercept")
    print(f"   • control_columns=['holiday_signal']")
    print(f"   • target_accept=0.90")
    
    print(f"\n🔍 例子2 (R² = 0.99) 特点:")
    print(f"   • 复杂DAG: 包含competitor_offers, market_growth等")
    print(f"   • 有time_varying_intercept=True")
    print(f"   • control_columns=['holiday_signal']")
    print(f"   • target_accept=0.95")
    print(f"   • 额外配置: intercept_tvp_config.ls_mu = 180")
    
    print(f"\n💡 关键差异:")
    print(f"   1. Time Varying Intercept 是最重要的差异")
    print(f"   2. 更复杂的DAG结构提供更多信息")
    print(f"   3. 更高的target_accept提高采样质量")

if __name__ == "__main__":
    try:
        print("🚀 开始不同R²值测试...")
        explain_notebook_differences()
        
        results = test_different_r2_values()
        
        print(f"\n{'='*60}")
        print("✅ 测试完成")
        print(f"{'='*60}")
        
        if len(results) >= 2:
            success = max(r['r2_score'] for r in results) - min(r['r2_score'] for r in results) > 0.05
            if success:
                print(f"🎯 成功：不同DAG现在产生显著不同的R²值！")
            else:
                print(f"⚠️ 部分成功：有差异但不够显著")
        else:
            print("❌ 测试失败：无法比较不同模型")
            
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc() 