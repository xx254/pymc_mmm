import React, { useCallback, useState, useRef, DragEvent } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  Position,
  NodeMouseHandler,
  EdgeMouseHandler,
  ReactFlowInstance,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './App.css';

// 节点类型定义
interface NodeTemplate {
  type: string;
  label: string;
  style: React.CSSProperties;
  description: string;
}

// DAG导出格式
interface DAGExport {
  nodes: Array<{
    id: string;
    label: string;
    type: string;
    position: { x: number; y: number };
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    style?: any;
  }>;
}

// 训练结果接口
interface TrainingResult {
  status: 'success' | 'error';
  message: string;
  model_summary?: any;
  convergence_info?: any;
  plots?: string[];
  errorDetails?: string;
  fullError?: string;
  error_details?: {
    error_type?: string;
    error_message?: string;
    traceback?: string;
    dag_type?: string;
    nodes_count?: number;
    edges_count?: number;
    causal_mmm_available?: boolean;
  };
}

const nodeTemplates: NodeTemplate[] = [
  {
    type: 'treatment',
    label: '治疗变量',
    style: {
      background: '#2196f3',
      color: 'white',
      border: '2px solid #1976d2',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    description: '营销渠道、广告投放等'
  },
  {
    type: 'outcome',
    label: '结果变量',
    style: {
      background: '#f44336',
      color: 'white',
      border: '2px solid #d32f2f',
      borderRadius: '8px',
      padding: '15px',
      fontSize: '14px',
      fontWeight: 'bold',
    },
    description: '销售额、转化率等'
  },
  {
    type: 'confounder',
    label: '混淆变量',
    style: {
      background: '#ff9800',
      color: 'white',
      border: '2px solid #f57c00',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    description: '季节性、竞争对手等'
  },
  {
    type: 'unobserved',
    label: '未观测变量',
    style: {
      background: '#9c27b0',
      color: 'white',
      border: '2px dashed #7b1fa2',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    description: '隐藏的影响因素'
  },
  {
    type: 'mediator',
    label: '中介变量',
    style: {
      background: '#4caf50',
      color: 'white',
      border: '2px solid #388e3c',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    description: '传递效应的中间变量'
  },
];

// DAG节点数据 - 来自CausalMMMTutorial
const businessScenarioNodes: Node[] = [
  {
    id: 'christmas',
    type: 'default',
    position: { x: 250, y: 0 },
    data: { label: 'Christmas (C)' },
    style: {
      background: '#ffeb3b',
      border: '2px dashed #f57f17',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
  },
  {
    id: 'x1',
    type: 'default',
    position: { x: 100, y: 150 },
    data: { label: 'X1 (社交媒体)' },
    style: {
      background: '#2196f3',
      color: 'white',
      border: '2px solid #1976d2',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
  },
  {
    id: 'x2',
    type: 'default',
    position: { x: 400, y: 150 },
    data: { label: 'X2 (搜索引擎)' },
    style: {
      background: '#9c27b0',
      color: 'white',
      border: '2px solid #7b1fa2',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
  },
  {
    id: 'competitor',
    type: 'default',
    position: { x: 550, y: 0 },
    data: { label: 'Competitor Offers (I)' },
    style: {
      background: '#ff9800',
      border: '2px dashed #f57c00',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
  },
  {
    id: 'market_growth',
    type: 'default',
    position: { x: 0, y: 0 },
    data: { label: 'Market Growth (G)' },
    style: {
      background: '#4caf50',
      color: 'white',
      border: '2px dashed #388e3c',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
  },
  {
    id: 'target',
    type: 'default',
    position: { x: 250, y: 300 },
    data: { label: 'Target (Sales)' },
    style: {
      background: '#f44336',
      color: 'white',
      border: '2px solid #d32f2f',
      borderRadius: '8px',
      padding: '15px',
      fontSize: '14px',
      fontWeight: 'bold',
    },
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
  },
];

const businessScenarioEdges: Edge[] = [
  {
    id: 'christmas-x1',
    source: 'christmas',
    target: 'x1',
    style: { stroke: '#f57f17', strokeWidth: 2, strokeDasharray: '5,5' },
    animated: false,
  },
  {
    id: 'christmas-x2',
    source: 'christmas',
    target: 'x2',
    style: { stroke: '#f57f17', strokeWidth: 2, strokeDasharray: '5,5' },
    animated: false,
  },
  {
    id: 'competitor-x2',
    source: 'competitor',
    target: 'x2',
    style: { stroke: '#f57c00', strokeWidth: 2, strokeDasharray: '5,5' },
    animated: false,
  },
  {
    id: 'x1-x2',
    source: 'x1',
    target: 'x2',
    style: { stroke: '#1976d2', strokeWidth: 2 },
    animated: true,
  },
  {
    id: 'christmas-target',
    source: 'christmas',
    target: 'target',
    style: { stroke: '#f57f17', strokeWidth: 2, strokeDasharray: '5,5' },
    animated: false,
  },
  {
    id: 'x1-target',
    source: 'x1',
    target: 'target',
    style: { stroke: '#1976d2', strokeWidth: 3 },
    animated: true,
  },
  {
    id: 'x2-target',
    source: 'x2',
    target: 'target',
    style: { stroke: '#7b1fa2', strokeWidth: 3 },
    animated: true,
  },
  {
    id: 'competitor-target',
    source: 'competitor',
    target: 'target',
    style: { stroke: '#f57c00', strokeWidth: 2, strokeDasharray: '5,5' },
    animated: false,
  },
  {
    id: 'market-target',
    source: 'market_growth',
    target: 'target',
    style: { stroke: '#388e3c', strokeWidth: 2, strokeDasharray: '5,5' },
    animated: false,
  },
];

// 简化版DAG
const simpleDagNodes: Node[] = [
  {
    id: 'x1_simple',
    type: 'default',
    position: { x: 100, y: 100 },
    data: { label: 'X1' },
    style: {
      background: '#2196f3',
      color: 'white',
      border: '2px solid #1976d2',
      borderRadius: '8px',
      padding: '10px',
    },
  },
  {
    id: 'x2_simple',
    type: 'default',
    position: { x: 300, y: 100 },
    data: { label: 'X2' },
    style: {
      background: '#9c27b0',
      color: 'white',
      border: '2px solid #7b1fa2',
      borderRadius: '8px',
      padding: '10px',
    },
  },
  {
    id: 'y_simple',
    type: 'default',
    position: { x: 200, y: 250 },
    data: { label: 'Y (销售额)' },
    style: {
      background: '#f44336',
      color: 'white',
      border: '2px solid #d32f2f',
      borderRadius: '8px',
      padding: '15px',
      fontWeight: 'bold',
    },
  },
];

const simpleDagEdges: Edge[] = [
  {
    id: 'x1-y-simple',
    source: 'x1_simple',
    target: 'y_simple',
    style: { stroke: '#1976d2', strokeWidth: 3 },
    animated: true,
  },
  {
    id: 'x2-y-simple',
    source: 'x2_simple',
    target: 'y_simple',
    style: { stroke: '#7b1fa2', strokeWidth: 3 },
    animated: true,
  },
];

let nodeId = 1000; // 用于生成新节点的ID

function App() {
  const [dagType, setDagType] = useState<'business' | 'simple' | 'custom'>('business');
  const [nodes, setNodes, onNodesChange] = useNodesState(businessScenarioNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(businessScenarioEdges);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [selectedEdges, setSelectedEdges] = useState<string[]>([]);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);
  const [showDAGExport, setShowDAGExport] = useState(false);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  const onConnect = useCallback(
    (params: Edge | Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onInit = (rfi: ReactFlowInstance) => setReactFlowInstance(rfi);

  const switchDAG = (type: 'business' | 'simple' | 'custom') => {
    setDagType(type);
    if (type === 'business') {
      setNodes(businessScenarioNodes);
      setEdges(businessScenarioEdges);
    } else if (type === 'simple') {
      setNodes(simpleDagNodes);
      setEdges(simpleDagEdges);
    } else if (type === 'custom') {
      setNodes([]);
      setEdges([]);
    }
    setSelectedNodes([]);
    setSelectedEdges([]);
    setTrainingResult(null);
  };

  // 导出DAG结构
  const exportDAG = (): DAGExport => {
    return {
      nodes: nodes.map(node => ({
        id: node.id,
        label: node.data.label,
        type: node.type || 'default',
        position: node.position
      })),
      edges: edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        style: edge.style
      }))
    };
  };

  // 生成Graphviz DOT格式的DAG字符串
  const generateDOTString = (): string => {
    let dotString = "digraph {\n";
    
    // 添加边（关系）
    edges.forEach(edge => {
      dotString += `  ${edge.source} -> ${edge.target};\n`;
    });
    
    dotString += "}";
    return dotString;
  };

  // 训练模型
  const trainModel = async () => {
    if (nodes.length === 0) {
      alert('请先创建DAG结构！');
      return;
    }

    setIsTraining(true);
    setTrainingResult(null);

    try {
      const dagExport = exportDAG();
      const dotString = generateDOTString();

      const response = await fetch('http://localhost:8000/train-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dag_structure: dagExport,
          dag_dot_string: dotString,
          dag_type: dagType
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: TrainingResult = await response.json();
      console.log('训练结果:', result); // 添加调试日志
      setTrainingResult(result);

    } catch (error) {
      console.error('训练模型时出错:', error);
      
      let errorMessage = '未知错误';
      let errorDetails = '';
      
      if (error instanceof TypeError && error.message.includes('fetch')) {
        errorMessage = '无法连接到后端服务器';
        errorDetails = '请确保API服务器正在运行 (python api_server.py)';
      } else if (error instanceof Error) {
        errorMessage = error.message;
        errorDetails = error.stack || '';
      }
      
      setTrainingResult({
        status: 'error',
        message: `训练失败: ${errorMessage}`,
        errorDetails: errorDetails,
        fullError: JSON.stringify(error, Object.getOwnPropertyNames(error))
      });
    } finally {
      setIsTraining(false);
    }
  };

  // 拖拽添加节点
  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');
      if (typeof type === 'undefined' || !type || !reactFlowInstance) {
        return;
      }

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const template = nodeTemplates.find(t => t.type === type);
      if (!template) return;

      const newNode: Node = {
        id: `${type}_${nodeId++}`,
        type: 'default',
        position,
        data: { label: `新${template.label}` },
        style: template.style,
        sourcePosition: Position.Bottom,
        targetPosition: Position.Top,
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  const onDragStart = (event: DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  // 选择节点
  const onNodeClick: NodeMouseHandler = useCallback((event, node) => {
    event.stopPropagation();
    setSelectedNodes([node.id]);
    setSelectedEdges([]);
  }, []);

  // 选择边
  const onEdgeClick: EdgeMouseHandler = useCallback((event, edge) => {
    event.stopPropagation();
    setSelectedEdges([edge.id]);
    setSelectedNodes([]);
  }, []);

  // 清除选择
  const onPaneClick = useCallback(() => {
    setSelectedNodes([]);
    setSelectedEdges([]);
  }, []);

  // 删除选中的元素
  const deleteSelected = useCallback(() => {
    if (selectedNodes.length > 0) {
      setNodes((nds) => nds.filter((node) => !selectedNodes.includes(node.id)));
      setEdges((eds) => eds.filter((edge) => 
        !selectedNodes.includes(edge.source) && !selectedNodes.includes(edge.target)
      ));
      setSelectedNodes([]);
    }
    if (selectedEdges.length > 0) {
      setEdges((eds) => eds.filter((edge) => !selectedEdges.includes(edge.id)));
      setSelectedEdges([]);
    }
  }, [selectedNodes, selectedEdges, setNodes, setEdges]);

  // 键盘事件处理
  const onKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.key === 'Delete' || event.key === 'Backspace') {
      deleteSelected();
    }
  }, [deleteSelected]);

  // 添加键盘监听
  React.useEffect(() => {
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [onKeyDown]);

  // 清空画布
  const clearCanvas = () => {
    setNodes([]);
    setEdges([]);
    setSelectedNodes([]);
    setSelectedEdges([]);
    setTrainingResult(null);
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* 侧边栏 */}
      <div style={{
        width: '300px',
        background: 'white',
        borderRight: '1px solid #ddd',
        padding: '20px',
        overflow: 'auto',
        boxShadow: '2px 0 4px rgba(0,0,0,0.1)'
      }}>
        <h3 style={{ margin: '0 0 20px 0', color: '#333', fontSize: '18px' }}>因果DAG编辑器</h3>
        
        {/* DAG模板选择 */}
        <div style={{ marginBottom: '25px' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>模板选择</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button
              onClick={() => switchDAG('business')}
              style={{
                padding: '8px 12px',
                backgroundColor: dagType === 'business' ? '#2196f3' : '#f5f5f5',
                color: dagType === 'business' ? 'white' : '#333',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              业务场景DAG
            </button>
            <button
              onClick={() => switchDAG('simple')}
              style={{
                padding: '8px 12px',
                backgroundColor: dagType === 'simple' ? '#2196f3' : '#f5f5f5',
                color: dagType === 'simple' ? 'white' : '#333',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              简化DAG
            </button>
            <button
              onClick={() => switchDAG('custom')}
              style={{
                padding: '8px 12px',
                backgroundColor: dagType === 'custom' ? '#2196f3' : '#f5f5f5',
                color: dagType === 'custom' ? 'white' : '#333',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              自定义DAG
            </button>
          </div>
        </div>

        {/* 节点工具箱 */}
        <div style={{ marginBottom: '25px' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>节点工具箱</h4>
          <p style={{ fontSize: '11px', color: '#888', margin: '0 0 15px 0' }}>
            拖拽下方节点到画布中
          </p>
          {nodeTemplates.map((template) => (
            <div
              key={template.type}
              draggable
              onDragStart={(event) => onDragStart(event, template.type)}
              style={{
                ...template.style,
                margin: '8px 0',
                cursor: 'grab',
                userSelect: 'none',
                textAlign: 'center',
                transition: 'transform 0.2s',
              }}
              onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.95)'}
              onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{template.label}</div>
              <div style={{ fontSize: '10px', opacity: 0.8, marginTop: '2px' }}>
                {template.description}
              </div>
            </div>
          ))}
        </div>

        {/* 操作按钮 */}
        <div style={{ marginBottom: '25px' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>操作</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button
              onClick={deleteSelected}
              disabled={selectedNodes.length === 0 && selectedEdges.length === 0}
              style={{
                padding: '8px 12px',
                backgroundColor: selectedNodes.length > 0 || selectedEdges.length > 0 ? '#f44336' : '#e0e0e0',
                color: selectedNodes.length > 0 || selectedEdges.length > 0 ? 'white' : '#999',
                border: 'none',
                borderRadius: '4px',
                cursor: selectedNodes.length > 0 || selectedEdges.length > 0 ? 'pointer' : 'not-allowed',
                fontSize: '12px'
              }}
            >
              删除选中 (Delete键)
            </button>
            <button
              onClick={clearCanvas}
              style={{
                padding: '8px 12px',
                backgroundColor: '#ff9800',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              清空画布
            </button>
          </div>
        </div>

        {/* DAG导出和模型训练 */}
        <div style={{ marginBottom: '25px' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>模型训练</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button
              onClick={() => setShowDAGExport(!showDAGExport)}
              style={{
                padding: '8px 12px',
                backgroundColor: '#4caf50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              {showDAGExport ? '隐藏' : '显示'} DAG结构
            </button>
            <button
              onClick={trainModel}
              disabled={isTraining || nodes.length === 0}
              style={{
                padding: '10px 12px',
                backgroundColor: isTraining ? '#9e9e9e' : '#2196f3',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: isTraining || nodes.length === 0 ? 'not-allowed' : 'pointer',
                fontSize: '12px',
                fontWeight: 'bold'
              }}
            >
              {isTraining ? '训练中...' : '🚀 训练因果模型'}
            </button>
          </div>

          {/* DAG导出显示 */}
          {showDAGExport && (
            <div style={{ marginTop: '10px' }}>
              <h5 style={{ margin: '0 0 5px 0', fontSize: '12px', color: '#666' }}>DOT格式:</h5>
              <textarea
                value={generateDOTString()}
                readOnly
                style={{
                  width: '100%',
                  height: '80px',
                  fontSize: '10px',
                  fontFamily: 'monospace',
                  padding: '5px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  resize: 'vertical'
                }}
              />
            </div>
          )}

          {/* 训练结果显示 */}
          {trainingResult && (
            <div style={{ 
              marginTop: '15px', 
              padding: '10px', 
              borderRadius: '4px',
              backgroundColor: trainingResult.status === 'success' ? '#e8f5e9' : '#ffebee',
              border: `1px solid ${trainingResult.status === 'success' ? '#4caf50' : '#f44336'}`
            }}>
              <h5 style={{ 
                margin: '0 0 5px 0', 
                fontSize: '12px', 
                color: trainingResult.status === 'success' ? '#2e7d32' : '#c62828' 
              }}>
                训练结果
              </h5>
              <p style={{ 
                margin: 0, 
                fontSize: '11px', 
                color: trainingResult.status === 'success' ? '#2e7d32' : '#c62828' 
              }}>
                {trainingResult.message}
              </p>
              {trainingResult.status === 'error' && (
                <div style={{ 
                  marginTop: '8px', 
                  fontSize: '10px', 
                  color: '#c62828',
                  backgroundColor: '#ffebee',
                  padding: '8px',
                  borderRadius: '4px',
                  border: '1px solid #ffcdd2'
                }}>
                  <strong>详细错误信息:</strong>
                  {trainingResult.error_details && (
                    <div style={{ marginTop: '5px' }}>
                      <div><strong>错误类型:</strong> {trainingResult.error_details.error_type}</div>
                      <div><strong>错误消息:</strong> {trainingResult.error_details.error_message}</div>
                      {trainingResult.error_details.dag_type && (
                        <div><strong>DAG类型:</strong> {trainingResult.error_details.dag_type}</div>
                      )}
                      {trainingResult.error_details.nodes_count !== undefined && (
                        <div><strong>节点数:</strong> {trainingResult.error_details.nodes_count}</div>
                      )}
                      {trainingResult.error_details.edges_count !== undefined && (
                        <div><strong>边数:</strong> {trainingResult.error_details.edges_count}</div>
                      )}
                      <div><strong>因果MMM可用:</strong> {trainingResult.error_details.causal_mmm_available ? '是' : '否'}</div>
                    </div>
                  )}
                  <details style={{ marginTop: '8px' }}>
                    <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>技术详情 (点击展开)</summary>
                    <pre style={{ 
                      fontSize: '8px', 
                      margin: '5px 0 0 0', 
                      whiteSpace: 'pre-wrap',
                      maxHeight: '200px',
                      overflow: 'auto',
                      background: 'white',
                      padding: '5px',
                      borderRadius: '3px'
                    }}>
                      {trainingResult.error_details?.traceback || JSON.stringify(trainingResult, null, 2)}
                    </pre>
                  </details>
                </div>
              )}
              {trainingResult.status === 'success' && trainingResult.model_summary && (
                <div style={{ marginTop: '8px', fontSize: '10px', color: '#666' }}>
                  <strong>模型信息:</strong>
                  <pre style={{ 
                    fontSize: '9px', 
                    margin: '5px 0 0 0', 
                    whiteSpace: 'pre-wrap',
                    maxHeight: '100px',
                    overflow: 'auto'
                  }}>
                    {JSON.stringify(trainingResult.model_summary, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 说明文档 */}
        <div style={{ fontSize: '11px', color: '#666', lineHeight: '1.5' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>使用说明</h4>
          <ul style={{ margin: 0, paddingLeft: '15px' }}>
            <li>拖拽节点到画布添加新节点</li>
            <li>拖拽节点边缘的圆点创建连接</li>
            <li>点击选中节点或边</li>
            <li>按Delete键删除选中元素</li>
            <li>可以拖拽移动节点位置</li>
            <li>设计完DAG后点击训练模型</li>
          </ul>
        </div>
      </div>

      {/* 主画布区域 */}
      <div ref={reactFlowWrapper} style={{ flex: 1 }}>
        <ReactFlow
          nodes={nodes.map(node => ({
            ...node,
            style: {
              ...node.style,
              boxShadow: selectedNodes.includes(node.id) ? '0 0 0 2px #2196f3' : 'none'
            }
          }))}
          edges={edges.map(edge => ({
            ...edge,
            style: {
              ...edge.style,
              stroke: selectedEdges.includes(edge.id) ? '#2196f3' : edge.style?.stroke,
              strokeWidth: selectedEdges.includes(edge.id) ? 3 : edge.style?.strokeWidth || 2
            }
          }))}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onInit={onInit}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          onPaneClick={onPaneClick}
          fitView
          style={{ background: '#f8f9fa' }}
        >
          <Controls />
          <MiniMap 
            style={{
              height: 120,
              backgroundColor: '#f8f9fa',
            }}
            zoomable
            pannable
          />
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
        </ReactFlow>
      </div>
    </div>
  );
}

export default App;
