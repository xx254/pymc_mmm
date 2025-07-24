#!/usr/bin/env python3
"""
测试不同DAG结构对模型结果的影响
分析为什么不同DAG会产生相同的R²值
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from causal_mmm_tutorial import CausalMMMTutorial
import numpy as np
import pandas as pd

def analyze_dag_differences():
    """分析不同DAG对模型的具体影响"""
    
    print("🔍 分析不同DAG结构对模型结果的影响")
    print("="*60)
    
    # 创建教程实例
    tutorial = CausalMMMTutorial()
    tutorial.generate_synthetic_data()
    
    # 定义三种不同的DAG结构
    test_dags = [
        {
            'name': 'Simple Direct',
            'dag_string': '''
            digraph {
                x1 -> y;
                x2 -> y;
            }
            ''',
            'description': '简单直接效应，无混淆变量'
        },
        {
            'name': 'With Confounders', 
            'dag_string': '''
            digraph {
                x1 -> y;
                x2 -> y;
                holiday_signal -> x1;
                holiday_signal -> x2;
                holiday_signal -> y;
            }
            ''',
            'description': '包含混淆变量的复杂结构'
        },
        {
            'name': 'Complex Interactions',
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
            'description': '包含变量间交互的复杂结构'
        }
    ]
    
    results = []
    
    for i, dag in enumerate(test_dags):
        print(f"\n📊 测试DAG {i+1}: {dag['name']}")
        print(f"描述: {dag['description']}")
        print("-" * 40)
        
        try:
            # 训练模型
            model_result = tutorial.run_causal_model(version="custom", custom_dag=dag['dag_string'])
            
            if model_result is not None:
                # 获取评估结果
                evaluation = tutorial.generate_model_evaluation_plots(model_result)
                
                if evaluation:
                    r2_score = evaluation['r2_score']
                    mape = evaluation['mape'] 
                    mae = evaluation['mae']
                    rmse = evaluation['rmse']
                    
                    print(f"✅ 训练成功!")
                    print(f"   R² Score: {r2_score:.6f}")
                    print(f"   MAPE: {mape:.6f}")
                    print(f"   MAE: {mae:.2f}")
                    print(f"   RMSE: {rmse:.2f}")
                    
                    # 分析模型内部结构
                    model_analysis = analyze_model_internals(model_result, dag['name'])
                    
                    results.append({
                        'name': dag['name'],
                        'r2_score': r2_score,
                        'mape': mape,
                        'mae': mae,
                        'rmse': rmse,
                        'model_analysis': model_analysis,
                        'dag_string': dag['dag_string']
                    })
                    
                else:
                    print("❌ 评估失败")
            else:
                print("❌ 训练失败")
                
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    # 分析结果
    print(f"\n{'='*60}")
    print("📈 结果对比分析")
    print(f"{'='*60}")
    
    if len(results) >= 2:
        # 比较R²值
        r2_values = [r['r2_score'] for r in results]
        r2_std = np.std(r2_values)
        r2_range = max(r2_values) - min(r2_values)
        
        print(f"\n🎯 R²分数对比:")
        for result in results:
            print(f"   {result['name']}: {result['r2_score']:.6f}")
        
        print(f"\n📊 统计分析:")
        print(f"   R²值标准差: {r2_std:.6f}")
        print(f"   R²值范围: {r2_range:.6f}")
        print(f"   平均R²: {np.mean(r2_values):.6f}")
        
        # 判断差异是否显著
        if r2_range < 0.001:
            print(f"\n⚠️  不同DAG产生了几乎相同的R²值!")
            print(f"   这表明DAG结构对预测准确性的影响很小")
            
            # 分析可能的原因
            print(f"\n🤔 可能的原因:")
            print(f"   1. 使用相同的底层数据和特征")
            print(f"   2. MMM模型的预测部分结构相同")
            print(f"   3. 不同DAG主要影响因果解释，而非预测准确性")
            print(f"   4. 当前样本大小和数据复杂度下，差异不够显著")
            
        else:
            print(f"\n✅ 不同DAG产生了不同的R²值!")
            print(f"   DAG结构确实影响了模型的预测性能")
    
    # 详细的模型内部分析
    print(f"\n🔬 模型内部结构分析:")
    for result in results:
        print(f"\n--- {result['name']} ---")
        analysis = result['model_analysis']
        for key, value in analysis.items():
            print(f"   {key}: {value}")
    
    return results

def analyze_model_internals(model_result, model_name):
    """分析模型内部结构和参数"""
    analysis = {}
    
    try:
        # 检查模型是否有因果图
        if hasattr(model_result, 'causal_graphical_model') and model_result.causal_graphical_model is not None:
            analysis['has_causal_constraints'] = True
            analysis['adjustment_set'] = str(model_result.causal_graphical_model.adjustment_set)
        else:
            analysis['has_causal_constraints'] = False
            analysis['adjustment_set'] = 'None'
        
        # 检查控制变量
        if hasattr(model_result, 'control_columns'):
            analysis['control_columns'] = model_result.control_columns
        else:
            analysis['control_columns'] = None
        
        # 检查模型参数数量
        if hasattr(model_result, 'idata') and model_result.idata is not None:
            try:
                posterior = model_result.idata.posterior
                analysis['parameter_count'] = len(posterior.data_vars)
                analysis['sample_shape'] = str(dict(posterior.dims))
            except:
                analysis['parameter_count'] = 'Unknown'
                analysis['sample_shape'] = 'Unknown'
        
        # 检查特征变换器
        if hasattr(model_result, 'target_transformer'):
            analysis['has_target_transformer'] = True
        else:
            analysis['has_target_transformer'] = False
            
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis

def compare_predictions_detail(results):
    """详细比较不同模型的预测差异"""
    if len(results) < 2:
        return
        
    print(f"\n🔍 预测详细对比:")
    
    # 这里可以加入更详细的预测对比逻辑
    # 比如比较不同时间点的预测差异
    pass

if __name__ == "__main__":
    try:
        results = analyze_dag_differences()
        
        print(f"\n{'='*60}")
        print("📝 总结")
        print(f"{'='*60}")
        
        if len(results) > 0:
            all_same = len(set(r['r2_score'] for r in results)) == 1
            if all_same:
                print("🎯 核心发现: 不同DAG结构产生了相同的R²值")
                print("\n💡 这是正常的，因为:")
                print("   • R²衡量预测准确性，而非因果解释能力")
                print("   • 不同DAG使用相同的特征和数据进行预测")
                print("   • DAG的主要作用是指导因果推断，而非提升预测性能")
                print("\n🔍 要看到DAG的真正价值，应该关注:")
                print("   • 渠道贡献的因果分解 (Channel Attribution)")
                print("   • 反事实分析 (Counterfactual Analysis)")  
                print("   • 混淆变量的控制效果")
                print("   • 因果效应的无偏估计")
            else:
                print("🎯 核心发现: 不同DAG结构产生了不同的R²值")
                print("   这表明DAG结构确实影响了模型性能")
        else:
            print("❌ 没有成功的模型结果可供分析")
            
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc() 