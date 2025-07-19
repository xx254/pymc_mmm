#!/usr/bin/env python3
"""
测试极端不同的DAG结构
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from causal_mmm_tutorial import CausalMMMTutorial
    print("✅ 成功导入 CausalMMMTutorial")
    
    # 创建实例
    tutorial = CausalMMMTutorial()
    print("✅ 成功创建 CausalMMMTutorial 实例")
    
    # 生成数据
    tutorial.generate_synthetic_data()
    print("✅ 成功生成合成数据")
    
    # 测试极端不同的DAG结构
    test_dags = [
        {
            "name": "仅X1模型",
            "dag_string": """
            digraph {
                x1 -> y;
            }
            """
        },
        {
            "name": "仅X2模型", 
            "dag_string": """
            digraph {
                x2 -> y;
            }
            """
        },
        {
            "name": "X1+X2模型",
            "dag_string": """
            digraph {
                x1 -> y;
                x2 -> y;
            }
            """
        },
        {
            "name": "X1+X2+假期模型",
            "dag_string": """
            digraph {
                x1 -> y;
                x2 -> y;
                holiday_signal -> y;
            }
            """
        }
    ]
    
    results = []
    
    for i, test_dag in enumerate(test_dags):
        print(f"\n{'='*50}")
        print(f"测试 {i+1}: {test_dag['name']}")
        print(f"{'='*50}")
        
        try:
            # 训练模型
            result = tutorial.run_causal_model(version="simple", custom_dag=test_dag['dag_string'])
            
            if result is not None:
                print("✅ 模型训练成功!")
                
                # 计算R²
                evaluation_result = tutorial.generate_model_evaluation_plots(result)
                
                if evaluation_result:
                    r2_score = evaluation_result['r2_score']
                    print(f"📈 R² Score: {r2_score:.4f}")
                    
                    results.append({
                        'name': test_dag['name'],
                        'r2_score': r2_score,
                        'dag_string': test_dag['dag_string']
                    })
                else:
                    print("❌ 模型评估失败")
            else:
                print("❌ 模型训练失败")
                
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    # 总结结果
    print(f"\n{'='*50}")
    print("测试结果总结")
    print(f"{'='*50}")
    
    for result in results:
        print(f"{result['name']}: R² = {result['r2_score']:.4f}")
    
    # 检查是否有差异
    r2_scores = [r['r2_score'] for r in results]
    if len(set(r2_scores)) > 1:
        print(f"\n✅ 不同DAG产生了不同的R²值!")
        print(f"R²值范围: {min(r2_scores):.4f} - {max(r2_scores):.4f}")
        print(f"最大差异: {max(r2_scores) - min(r2_scores):.4f}")
        
        # 分析差异
        print(f"\n📊 差异分析:")
        for i, result in enumerate(results):
            if i > 0:
                diff = result['r2_score'] - results[i-1]['r2_score']
                print(f"  {result['name']} vs {results[i-1]['name']}: {diff:+.4f}")
    else:
        print(f"\n⚠️ 所有DAG产生了相同的R²值: {r2_scores[0]:.4f}")
        
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc() 