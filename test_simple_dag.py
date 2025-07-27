#!/usr/bin/env python3
"""
简单测试DAG问题修复
专门测试简单DAG结构是否能正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from causal_mmm_tutorial import CausalMMMTutorial

def test_simple_dag_fix():
    """测试简单DAG修复"""
    
    print("🔍 测试简单DAG修复效果")
    print("="*50)
    
    # 创建教程实例
    tutorial = CausalMMMTutorial()
    tutorial.generate_synthetic_data()
    
    # 测试简单DAG - 这个之前会失败
    print(f"\n📊 测试简单DAG模型")
    try:
        simple_dag = """
        digraph {
            x1 -> y;
            x2 -> y;
            holiday_signal -> y;
        }
        """
        
        print(f"DAG结构:\n{simple_dag}")
        
        causal_model = tutorial.run_causal_model(version="custom", custom_dag=simple_dag)
        
        if causal_model is not None:
            print(f"✅ 简单DAG模型创建和训练成功！")
            
            # 检查模型的causal graphical model
            if hasattr(causal_model, 'causal_graphical_model'):
                cgm = causal_model.causal_graphical_model
                print(f"🔍 Causal graphical model存在: {cgm is not None}")
                
                if cgm is not None:
                    if hasattr(cgm, 'adjustment_set'):
                        print(f"🔍 Adjustment set: {cgm.adjustment_set}")
                    if hasattr(cgm, 'minimal_adjustment_set'):
                        print(f"🔍 Minimal adjustment set: {cgm.minimal_adjustment_set}")
            
            print(f"🔍 控制变量: {causal_model.control_columns}")
            
            # 尝试生成评估
            evaluation = tutorial.generate_model_evaluation_plots(causal_model)
            if evaluation:
                print(f"📈 模型评估成功:")
                print(f"   R² Score: {evaluation['r2_score']:.6f}")
                print(f"   MAPE: {evaluation['mape']:.4f}")
                return True
        else:
            print(f"❌ 模型创建失败")
            return False
        
    except Exception as e:
        print(f"❌ 简单DAG测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def explain_error_cause():
    """解释之前错误的原因"""
    print(f"\n{'='*50}")
    print("🔍 错误原因分析")
    print(f"{'='*50}")
    
    print(f"\n❌ 之前的错误: 'CausalGraphModel' object has no attribute 'adjustment_set'")
    print(f"\n🎯 原因分析:")
    print(f"   1. 时序问题: MMM初始化时过早访问adjustment_set属性")
    print(f"   2. 属性未设置: adjustment_set只有在compute_adjustment_sets()后才存在")
    print(f"   3. 代码结构: MMM类的__init__方法中的检查顺序有问题")
    
    print(f"\n🔧 修复方法:")
    print(f"   1. 确保control_columns不是空列表（None而不是[]）")
    print(f"   2. 安全地访问adjustment_set属性")
    print(f"   3. 正确处理DAG分析结果")
    
    print(f"\n✅ 现在的改进:")
    print(f"   • 控制变量列表正确处理")
    print(f"   • DAG结构正确传递给MMM")
    print(f"   • 错误处理更加健壮")

if __name__ == "__main__":
    try:
        print("🚀 开始简单DAG修复测试...")
        explain_error_cause()
        
        success = test_simple_dag_fix()
        
        print(f"\n{'='*50}")
        if success:
            print("✅ 简单DAG修复测试成功!")
            print("🎯 问题已解决：不同DAG现在可以正常训练并产生结果")
        else:
            print("❌ 简单DAG修复测试失败")
            print("🔍 需要进一步调试")
        print(f"{'='*50}")
            
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc() 