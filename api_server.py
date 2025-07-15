#!/usr/bin/env python3
"""
FastAPI后端服务，用于因果DAG编辑器
连接React前端和Python的因果MMM模型训练
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import asyncio
import traceback
import logging
import sys
import os

# 添加当前目录到Python路径，以便导入causal_mmm_tutorial
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入CausalMMMTutorial
try:
    from causal_mmm_tutorial import CausalMMMTutorial
    CAUSAL_MMM_AVAILABLE = True
    print("✅ CausalMMMTutorial导入成功")
except ImportError as e:
    print(f"⚠️  警告: 无法导入CausalMMMTutorial: {e}")
    CAUSAL_MMM_AVAILABLE = False
    
    # 创建一个备用的基础类
    class CausalMMMTutorial:
        def __init__(self):
            self.df = None
            self.data = None
        
        def generate_synthetic_data(self):
            print("模拟数据生成（备用模式）")
            return None
        
        def run_causal_model(self, version="full"):
            print(f"模拟因果模型训练（备用模式，版本：{version}）")
            return None
        
        def run_correlational_model(self):
            print("模拟相关性模型训练（备用模式）")
            return None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="因果DAG编辑器API", version="1.0.0")

# 添加CORS中间件以允许React前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React开发服务器
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型定义
class NodeData(BaseModel):
    id: str
    label: str
    type: str
    position: Dict[str, float]

class EdgeData(BaseModel):
    id: str
    source: str
    target: str
    style: Optional[Dict[str, Any]] = None

class DAGStructure(BaseModel):
    nodes: List[NodeData]
    edges: List[EdgeData]

class TrainingRequest(BaseModel):
    dag_structure: DAGStructure
    dag_dot_string: str
    dag_type: str

class TrainingResponse(BaseModel):
    status: str
    message: str
    model_summary: Optional[Dict[str, Any]] = None
    convergence_info: Optional[Dict[str, Any]] = None
    plots: Optional[List[str]] = None

class EnhancedCausalMMMTutorial(CausalMMMTutorial):
    """增强版的CausalMMMTutorial，支持动态DAG"""
    
    def __init__(self):
        super().__init__()
        self.custom_dag = None
        self.custom_dag_dot = None
        
    def set_custom_dag(self, dag_structure: DAGStructure, dag_dot_string: str):
        """设置自定义DAG结构"""
        self.custom_dag = dag_structure
        self.custom_dag_dot = dag_dot_string
        
    def create_dynamic_dag_string(self, dag_structure: DAGStructure) -> str:
        """根据DAG结构创建DOT字符串"""
        if not dag_structure.edges:
            return "digraph { }"
            
        dot_string = "digraph {\n"
        
        # 添加节点定义（可选，用于更好的可视化）
        for node in dag_structure.nodes:
            node_id = node.id.replace(' ', '_').replace('(', '').replace(')', '')
            dot_string += f'  {node_id} [label="{node.label}"];\n'
        
        # 添加边
        for edge in dag_structure.edges:
            source = edge.source.replace(' ', '_').replace('(', '').replace(')', '')
            target = edge.target.replace(' ', '_').replace('(', '').replace(')', '')
            dot_string += f"  {source} -> {target};\n"
        
        dot_string += "}"
        return dot_string
        
    def map_dag_to_model_variables(self, dag_structure: DAGStructure) -> Dict[str, Any]:
        """将DAG结构映射到模型变量"""
        # 识别不同类型的节点
        treatment_nodes = []
        outcome_nodes = []
        control_nodes = []
        
        for node in dag_structure.nodes:
            node_label = node.label.lower()
            node_id = node.id.lower()
            
            # 根据节点标签和ID识别节点类型
            if any(keyword in node_label for keyword in ['x1', 'x2', '社交', '搜索', '营销', '广告', '治疗']):
                treatment_nodes.append(node.id)
            elif any(keyword in node_label for keyword in ['target', 'sales', '销售', 'y', '目标', '结果']):
                outcome_nodes.append(node.id)
            elif any(keyword in node_label for keyword in ['christmas', 'holiday', 'competitor', 'market', '假期', '竞争', '市场', '混淆', '未观测', '中介']):
                control_nodes.append(node.id)
                
        return {
            'treatment_nodes': treatment_nodes,
            'outcome_nodes': outcome_nodes,
            'control_nodes': control_nodes,
            'channel_columns': [node for node in treatment_nodes if any(x in node.lower() for x in ['x1', 'x2'])],
            'outcome_node': outcome_nodes[0] if outcome_nodes else 'y'
        }
        
    def run_custom_model(self, dag_structure: DAGStructure, dag_type: str):
        """根据自定义DAG运行模型"""
        try:
            print(f"🔥 DEBUG: 开始训练自定义模型，DAG类型: {dag_type}")
            print(f"🔥 DEBUG: CAUSAL_MMM_AVAILABLE = {CAUSAL_MMM_AVAILABLE}")
            logger.info(f"开始训练自定义模型，DAG类型: {dag_type}")
            
            # 设置自定义DAG
            self.set_custom_dag(dag_structure, self.create_dynamic_dag_string(dag_structure))
            
            # 映射DAG到模型变量
            model_mapping = self.map_dag_to_model_variables(dag_structure)
            print(f"🔥 DEBUG: 模型映射: {model_mapping}")
            logger.info(f"模型映射: {model_mapping}")
            
            if not CAUSAL_MMM_AVAILABLE:
                # 如果真实的PyMC-Marketing不可用，返回模拟结果
                print("🔥 DEBUG: 使用模拟模式训练模型...")
                logger.info("使用模拟模式训练模型...")
                
                model_summary = {
                    'dag_type': dag_type,
                    'nodes_count': len(dag_structure.nodes),
                    'edges_count': len(dag_structure.edges),
                    'treatment_variables': model_mapping['treatment_nodes'],
                    'outcome_variables': model_mapping['outcome_nodes'],
                    'control_variables': model_mapping['control_nodes'],
                    'mode': 'simulation'
                }
                
                print("🔥 DEBUG: 返回模拟成功结果")
                return {
                    'status': 'success',
                    'message': f'模型训练完成（模拟模式）！使用了{len(dag_structure.nodes)}个节点和{len(dag_structure.edges)}个边的DAG结构。注意：这是模拟结果，请安装完整的依赖包以获得真实的模型训练结果。',
                    'model_summary': model_summary,
                    'convergence_info': {
                        'r_hat_max': 1.01,  # 模拟的收敛指标
                        'ess_bulk_min': 1000,
                        'divergences': 0,
                        'mode': 'simulated'
                    }
                }
            
            print("🔥 DEBUG: 尝试真实模型训练...")
            # 生成数据（如果还没有生成）
            if self.df is None:
                print("🔥 DEBUG: 生成合成数据...")
                logger.info("生成合成数据...")
                try:
                    self.generate_synthetic_data()
                    print("🔥 DEBUG: 数据生成成功")
                except Exception as e:
                    print(f"🔥 DEBUG: 数据生成失败: {e}")
                    print(f"🔥 DEBUG: 数据生成失败详细信息: {traceback.format_exc()}")
                    raise
            
            # 根据DAG类型选择训练方法
            print(f"🔥 DEBUG: 开始根据DAG类型 {dag_type} 训练模型...")
            result = None
            if dag_type == 'business':
                # 使用预定义的业务场景模型
                print("🔥 DEBUG: 运行业务场景模型...")
                try:
                    result = self.run_causal_model(version="full")
                    print(f"🔥 DEBUG: 业务场景模型训练结果: {type(result)}")
                except Exception as e:
                    error_msg = f"业务场景模型训练失败: {str(e)}"
                    error_traceback = traceback.format_exc()
                    print(f"🔥 DEBUG: {error_msg}")
                    print(f"🔥 DEBUG: 异常类型: {type(e).__name__}")
                    print(f"🔥 DEBUG: 详细错误: {error_traceback}")
                    logger.error(error_msg)
                    logger.error(f"异常类型: {type(e).__name__}")
                    logger.error(error_traceback)
                    # 重新抛出异常，但添加更多上下文信息
                    raise Exception(f"{error_msg} (异常类型: {type(e).__name__})") from e
            elif dag_type == 'simple':
                # 使用简化模型
                print("🔥 DEBUG: 运行简化模型...")
                try:
                    result = self.run_causal_model(version="simple")
                    print(f"🔥 DEBUG: 简化模型训练结果: {type(result)}")
                except Exception as e:
                    error_msg = f"简化模型训练失败: {str(e)}"
                    error_traceback = traceback.format_exc()
                    print(f"🔥 DEBUG: {error_msg}")
                    print(f"🔥 DEBUG: 异常类型: {type(e).__name__}")
                    print(f"🔥 DEBUG: 详细错误: {error_traceback}")
                    logger.error(error_msg)
                    logger.error(f"异常类型: {type(e).__name__}")
                    logger.error(error_traceback)
                    # 重新抛出异常，但添加更多上下文信息
                    raise Exception(f"{error_msg} (异常类型: {type(e).__name__})") from e
            else:
                # 自定义模型 - 使用基础的相关性模型
                print("🔥 DEBUG: 运行自定义模型（基础相关性模型）...")
                logger.info("运行自定义模型（基础相关性模型）...")
                try:
                    result = self.run_correlational_model()
                    print(f"🔥 DEBUG: 相关性模型训练结果: {type(result)}")
                except Exception as e:
                    error_msg = f"相关性模型训练失败: {str(e)}"
                    error_traceback = traceback.format_exc()
                    print(f"🔥 DEBUG: {error_msg}")
                    print(f"🔥 DEBUG: 异常类型: {type(e).__name__}")
                    print(f"🔥 DEBUG: 详细错误: {error_traceback}")
                    logger.error(error_msg)
                    logger.error(f"异常类型: {type(e).__name__}")
                    logger.error(error_traceback)
                    # 重新抛出异常，但添加更多上下文信息
                    raise Exception(f"{error_msg} (异常类型: {type(e).__name__})") from e
            
            if result is None:
                error_msg = "模型训练结果为None - 可能的原因：数据问题、模型配置错误、或依赖包不完整"
                print(f"🔥 DEBUG: {error_msg}")
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': f'PyMC-Marketing训练失败: {error_msg}',
                    'error_details': {
                        'error_type': 'NullResult',
                        'dag_type': dag_type,
                        'nodes_count': len(dag_structure.nodes),
                        'edges_count': len(dag_structure.edges),
                        'causal_mmm_available': CAUSAL_MMM_AVAILABLE
                    }
                }
            
            print(f"🔥 DEBUG: 模型训练成功，结果类型: {type(result)}")
            # 准备返回结果
            model_summary = {
                'dag_type': dag_type,
                'nodes_count': len(dag_structure.nodes),
                'edges_count': len(dag_structure.edges),
                'treatment_variables': model_mapping['treatment_nodes'],
                'outcome_variables': model_mapping['outcome_nodes'],
                'control_variables': model_mapping['control_nodes']
            }
            
            # 检查模型收敛性
            convergence_info = {}
            if hasattr(result, 'idata') and result.idata is not None:
                try:
                    import arviz as az
                    convergence_info = {
                        'r_hat_max': float(az.rhat(result.idata).max()),
                        'ess_bulk_min': float(az.ess(result.idata).min()),
                        'divergences': int(result.idata["sample_stats"]["diverging"].sum())
                    }
                    print(f"🔥 DEBUG: 收敛性指标计算成功: {convergence_info}")
                except Exception as e:
                    print(f"🔥 DEBUG: 无法计算收敛性指标: {e}")
                    logger.warning(f"无法计算收敛性指标: {e}")
            else:
                print("🔥 DEBUG: 模型结果没有idata属性")
            
            print("🔥 DEBUG: 返回成功结果")
            return {
                'status': 'success',
                'message': f'模型训练完成！使用了{len(dag_structure.nodes)}个节点和{len(dag_structure.edges)}个边的DAG结构。',
                'model_summary': model_summary,
                'convergence_info': convergence_info
            }
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"🔥 DEBUG: 模型训练发生异常: {str(e)}")
            print(f"🔥 DEBUG: 异常类型: {type(e).__name__}")
            print(f"🔥 DEBUG: 详细异常信息: {error_traceback}")
            logger.error(f"模型训练失败: {str(e)}")
            logger.error(f"异常类型: {type(e).__name__}")
            logger.error(error_traceback)
            
            # 构建详细的错误响应
            error_response = {
                'status': 'error',
                'message': f'模型训练失败: {str(e)}',
                'error_details': {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'traceback': error_traceback,
                    'dag_type': dag_type if 'dag_type' in locals() else 'unknown',
                    'causal_mmm_available': CAUSAL_MMM_AVAILABLE
                }
            }
            
            # 如果有DAG结构信息，也包含进去
            if 'dag_structure' in locals() and dag_structure:
                error_response['error_details'].update({
                    'nodes_count': len(dag_structure.nodes),
                    'edges_count': len(dag_structure.edges)
                })
            
            return error_response

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "因果DAG编辑器API",
        "version": "1.0.0",
        "causal_mmm_available": CAUSAL_MMM_AVAILABLE,
        "status": "运行中" if CAUSAL_MMM_AVAILABLE else "模拟模式",
        "endpoints": {
            "train_model": "/train-model",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "causal_mmm_available": CAUSAL_MMM_AVAILABLE,
        "mode": "full" if CAUSAL_MMM_AVAILABLE else "simulation"
    }

@app.post("/train-model", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """训练因果模型的主要端点"""
    
    try:
        logger.info(f"收到训练请求，DAG类型: {request.dag_type}")
        logger.info(f"节点数量: {len(request.dag_structure.nodes)}")
        logger.info(f"边数量: {len(request.dag_structure.edges)}")
        
        # 验证输入
        if len(request.dag_structure.nodes) == 0:
            raise HTTPException(
                status_code=400,
                detail="DAG结构不能为空，请至少添加一个节点"
            )
        
        # 创建增强版教程实例
        tutorial = EnhancedCausalMMMTutorial()
        
        # 运行模型训练
        result = tutorial.run_custom_model(request.dag_structure, request.dag_type)
        
        return TrainingResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"训练模型时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"内部服务器错误: {str(e)}"
        )

@app.post("/validate-dag")
async def validate_dag(dag_structure: DAGStructure):
    """验证DAG结构的有效性"""
    try:
        # 基本验证
        if len(dag_structure.nodes) == 0:
            return {"valid": False, "message": "DAG必须包含至少一个节点"}
        
        # 检查边的有效性
        node_ids = {node.id for node in dag_structure.nodes}
        for edge in dag_structure.edges:
            if edge.source not in node_ids:
                return {"valid": False, "message": f"边的源节点 '{edge.source}' 不存在"}
            if edge.target not in node_ids:
                return {"valid": False, "message": f"边的目标节点 '{edge.target}' 不存在"}
        
        # 检查是否有环（简单检查）
        # TODO: 实现更复杂的环检测算法
        
        return {"valid": True, "message": "DAG结构有效"}
        
    except Exception as e:
        logger.error(f"验证DAG时发生错误: {str(e)}")
        return {"valid": False, "message": f"验证失败: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 启动因果DAG编辑器API服务...")
    print("📊 前端地址: http://localhost:3000")
    print("🔗 API地址: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/docs")
    print(f"🔧 模式: {'完整功能' if CAUSAL_MMM_AVAILABLE else '模拟模式'}")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 