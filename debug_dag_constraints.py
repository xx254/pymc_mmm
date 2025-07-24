#!/usr/bin/env python3
"""
诊断DAG约束是否真正生效的脚本
验证为什么不同DAG产生相同的R²
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from causal_mmm_tutorial import CausalMMMTutorial
import numpy as np
import pandas as pd

def diagnose_dag_constraints():
    """诊断DAG约束是否真正应用"""
    
    print("🔬 诊断DAG约束应用情况")
    print("="*60)
    
    # 创建教程实例
    tutorial = CausalMMMTutorial()
    tutorial.generate_synthetic_data()
    
    print(f"📊 数据信息:")
    print(f"   数据形状: {tutorial.data.shape}")
    print(f"   特征列: {list(tutorial.data.columns)}")
    print(f"   目标变量统计: {tutorial.data['y'].describe()}")
    
    # 检查每列的方差
    print(f"\n📈 各列方差检查:")
    for col in tutorial.data.columns:
        if col != 'y':
            variance = tutorial.data[col].var()
            print(f"   {col}: {variance:.8f} (是否>1e-10: {variance > 1e-10})")
    
    # 测试三种不同的DAG
    test_cases = [
        {
            'name': 'Simple DAG (无控制变量)',
            'dag_string': '''
            digraph {
                x1 -> y;
                x2 -> y;
            }
            ''',
            'expected_control_vars': []
        },
        {
            'name': 'With Holiday Control (有控制变量)',
            'dag_string': '''
            digraph {
                x1 -> y;
                x2 -> y;
                holiday_signal -> x1;
                holiday_signal -> x2;
                holiday_signal -> y;
            }
            ''',
            'expected_control_vars': ['holiday_signal']
        },
        {
            'name': 'Complex DAG (复杂控制)',
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
            'expected_control_vars': ['holiday_signal']
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"🧪 测试案例 {i+1}: {test_case['name']}")
        print(f"{'='*60}")
        
        # 手动调用控制变量提取逻辑
        print(f"📋 DAG字符串:\n{test_case['dag_string']}")
        
        try:
            # 测试控制变量提取
            extracted_controls = tutorial._extract_control_variables_from_dag(test_case['dag_string'])
            print(f"🔍 提取的控制变量: {extracted_controls}")
            print(f"🎯 期望的控制变量: {test_case['expected_control_vars']}")
            
            # 检查控制变量是否存在于数据中
            available_controls = []
            for var in extracted_controls:
                if var in tutorial.data.columns:
                    variance = tutorial.data[var].var()
                    print(f"   ✅ {var} 存在于数据中，方差: {variance:.8f}")
                    if variance > 1e-10:
                        available_controls.append(var)
                        print(f"      ✅ 方差检查通过")
                    else:
                        print(f"      ❌ 方差过小，将被忽略")
                else:
                    print(f"   ❌ {var} 不存在于数据中")
            
            print(f"🔧 最终可用的控制变量: {available_controls}")
            
            # 训练模型并检查是否真正应用了DAG约束
            print(f"\n🚀 开始训练模型...")
            model_result = tutorial.run_causal_model(version="custom", custom_dag=test_case['dag_string'])
            
            if model_result is not None:
                # 检查模型是否真正应用了因果约束
                has_causal_constraints = False
                adjustment_set = None
                
                if hasattr(model_result, 'causal_graphical_model') and model_result.causal_graphical_model is not None:
                    has_causal_constraints = True
                    adjustment_set = model_result.causal_graphical_model.adjustment_set
                    print(f"✅ 模型应用了因果约束")
                    print(f"   调整集: {adjustment_set}")
                else:
                    print(f"❌ 模型没有应用因果约束（回退到标准MMM）")
                
                # 获取R²分数
                evaluation = tutorial.generate_model_evaluation_plots(model_result)
                r2_score = evaluation['r2_score'] if evaluation else None
                
                results.append({
                    'name': test_case['name'],
                    'dag_string': test_case['dag_string'],
                    'extracted_controls': extracted_controls,
                    'available_controls': available_controls,
                    'has_causal_constraints': has_causal_constraints,
                    'adjustment_set': adjustment_set,
                    'r2_score': r2_score
                })
                
                print(f"📈 R² Score: {r2_score:.6f}")
                
            else:
                print(f"❌ 模型训练失败")
                results.append({
                    'name': test_case['name'],
                    'dag_string': test_case['dag_string'],
                    'extracted_controls': extracted_controls,
                    'available_controls': available_controls,
                    'has_causal_constraints': False,
                    'adjustment_set': None,
                    'r2_score': None
                })
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 分析结果
    print(f"\n{'='*60}")
    print("📊 诊断结果分析")
    print(f"{'='*60}")
    
    print(f"\n🔍 DAG约束应用情况:")
    for result in results:
        constraint_status = "✅ 应用" if result['has_causal_constraints'] else "❌ 未应用"
        print(f"   {result['name']}: {constraint_status}")
        if result['adjustment_set']:
            print(f"      调整集: {result['adjustment_set']}")
    
    print(f"\n📈 R²分数对比:")
    r2_scores = []
    for result in results:
        if result['r2_score'] is not None:
            r2_scores.append(result['r2_score'])
            print(f"   {result['name']}: {result['r2_score']:.6f}")
    
    if len(r2_scores) >= 2:
        r2_std = np.std(r2_scores)
        r2_range = max(r2_scores) - min(r2_scores)
        
        print(f"\n📊 R²统计:")
        print(f"   标准差: {r2_std:.6f}")
        print(f"   范围: {r2_range:.6f}")
        
        if r2_range < 0.001:
            print(f"\n🚨 发现问题: R²值几乎相同!")
            
            # 分析原因
            causal_models = [r for r in results if r['has_causal_constraints']]
            non_causal_models = [r for r in results if not r['has_causal_constraints']]
            
            print(f"\n🔍 原因分析:")
            print(f"   应用了DAG约束的模型数量: {len(causal_models)}")
            print(f"   未应用DAG约束的模型数量: {len(non_causal_models)}")
            
            if len(non_causal_models) == len(results):
                print(f"   💡 根本原因: 所有模型都回退到了标准MMM，没有真正应用DAG约束！")
                print(f"   💡 这解释了为什么不同DAG产生相同的R²值")
            elif len(causal_models) > 0 and len(non_causal_models) > 0:
                print(f"   💡 部分应用了DAG约束，部分没有")
            else:
                print(f"   💡 所有模型都应用了DAG约束，但R²仍然相同")
                print(f"   💡 这可能表明当前数据集中DAG差异对预测性能影响很小")
        else:
            print(f"\n✅ R²值有明显差异，DAG约束正常工作")
    
    return results

def test_control_variable_extraction():
    """测试控制变量提取逻辑"""
    print(f"\n🧪 测试控制变量提取逻辑")
    print("-" * 40)
    
    tutorial = CausalMMMTutorial()
    
    test_dags = [
        "digraph { x1 -> y; x2 -> y; }",
        "digraph { x1 -> y; x2 -> y; holiday_signal -> y; }",
        "digraph { x1 -> y; x2 -> y; holiday_signal -> x1; holiday_signal -> y; }",
    ]
    
    for i, dag in enumerate(test_dags):
        print(f"\nDAG {i+1}: {dag}")
        try:
            controls = tutorial._extract_control_variables_from_dag(dag)
            print(f"提取的控制变量: {controls}")
        except Exception as e:
            print(f"提取失败: {e}")

if __name__ == "__main__":
    try:
        # 先测试控制变量提取逻辑
        test_control_variable_extraction()
        
        # 然后进行完整诊断
        results = diagnose_dag_constraints()
        
        print(f"\n{'='*60}")
        print("🎯 核心结论")
        print(f"{'='*60}")
        
        print("你的理解是完全正确的：")
        print("✅ 理论上，不同的DAG应该产生不同的R²")
        print("✅ R²越接近1，说明DAG越接近真实的因果结构")
        print()
        print("问题在于当前实现中：")
        print("🔧 DAG约束可能没有真正应用到模型中")
        print("🔧 系统可能回退到了标准的MMM模型")
        print("🔧 这导致所有模型实际上都在做相同的预测任务")
        print()
        print("要验证DAG的真正效果，需要确保：")
        print("1. 控制变量被正确识别和包含")
        print("2. DAG约束真正应用到模型结构中")
        print("3. 不同DAG产生不同的调整集（adjustment sets）")
        
    except Exception as e:
        print(f"❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc() 