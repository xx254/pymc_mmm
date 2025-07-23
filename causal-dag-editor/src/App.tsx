import React, { CSSProperties, useCallback, useState, useRef, DragEvent } from 'react';
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
  MarkerType,
} from 'reactflow';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement,
  Filler
} from 'chart.js';
import { Line, Scatter, Bar } from 'react-chartjs-2';

import 'reactflow/dist/style.css';
import './App.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement,
  Filler
);

// Node type definition
interface NodeTemplate {
  type: string;
  label: string;
  style: React.CSSProperties;
  description: string;
}

// DAG export format
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

// Training result interface
interface TrainingResult {
  status: 'success' | 'error';
  message: string;
  model_summary?: {
    dag_type?: string;
    model_type?: 'causal' | 'correlational';
    has_causal_constraints?: boolean;
    nodes_count?: number;
    edges_count?: number;
    treatment_variables?: string[];
    outcome_variables?: string[];
    control_variables?: string[];
    fit_quality?: {
      r2_score?: number;
      mape?: number;
      mae?: number;
      rmse?: number;
      sample_size?: number;
    };
    data_info?: {
      prediction_mean?: number;
      prediction_std?: number;
      actual_mean?: number;
      actual_std?: number;
    };
    plot_available?: boolean;
    plot_path?: string;
    chart_data?: {
      actual_vs_predicted?: {
        actual: number[];
        predicted: number[];
      };
      time_series?: {
        dates: (string | number)[];
        actual: number[];
        predicted: number[];
        confidence_upper: number[];
        confidence_lower: number[];
      };
      media_contribution?: {
        dates: (string | number)[];
        x1: {
          predicted_mean: number[];
          confidence_upper: number[];
          confidence_lower: number[];
          real_effect: number[];
        };
        x2: {
          predicted_mean: number[];
          confidence_upper: number[];
          confidence_lower: number[];
          real_effect: number[];
        };
      };
    };
  };
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
    label: 'Treatment',
    style: {
      background: '#2196f3',
      color: 'white',
      border: '2px solid #1976d2',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    description: 'Marketing channels, ad campaigns, etc.'
  },
  {
    type: 'outcome',
    label: 'Outcome',
    style: {
      background: '#f44336',
      color: 'white',
      border: '2px solid #d32f2f',
      borderRadius: '8px',
      padding: '15px',
      fontSize: '14px',
      fontWeight: 'bold',
    },
    description: 'Sales, conversion rate, etc.'
  },
  {
    type: 'confounder',
    label: 'Confounder',
    style: {
      background: '#ff9800',
      color: 'white',
      border: '2px solid #f57c00',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    description: 'Seasonality, competitors, etc.'
  },
  {
    type: 'unobserved',
    label: 'Unobserved',
    style: {
      background: '#9c27b0',
      color: 'white',
      border: '2px dashed #7b1fa2',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    description: 'Hidden confounding factors'
  },
  {
    type: 'mediator',
    label: 'Mediator',
    style: {
      background: '#4caf50',
      color: 'white',
      border: '2px solid #388e3c',
      borderRadius: '8px',
      padding: '10px',
      fontSize: '12px',
      fontWeight: 'bold',
    },
    description: 'Intermediate variables for effects'
  },
];

// DAG node data - from CausalMMMTutorial
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
    data: { label: 'X1 (Social Media)' },
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
    data: { label: 'X2 (Search Engine)' },
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
    style: { stroke: '#222', strokeWidth: 2, strokeDasharray: '5,5' } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
  {
    id: 'christmas-x2',
    source: 'christmas',
    target: 'x2',
    style: { stroke: '#222', strokeWidth: 2, strokeDasharray: '5,5' } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
  {
    id: 'competitor-x2',
    source: 'competitor',
    target: 'x2',
    style: { stroke: '#222', strokeWidth: 2, strokeDasharray: '5,5' } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
  {
    id: 'x1-x2',
    source: 'x1',
    target: 'x2',
    style: { stroke: '#222', strokeWidth: 2 } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
  {
    id: 'christmas-target',
    source: 'christmas',
    target: 'target',
    style: { stroke: '#222', strokeWidth: 2, strokeDasharray: '5,5' } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
  {
    id: 'x1-target',
    source: 'x1',
    target: 'target',
    style: { stroke: '#222', strokeWidth: 3 } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
  {
    id: 'x2-target',
    source: 'x2',
    target: 'target',
    style: { stroke: '#222', strokeWidth: 3 } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
  {
    id: 'competitor-target',
    source: 'competitor',
    target: 'target',
    style: { stroke: '#222', strokeWidth: 2, strokeDasharray: '5,5' } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
  {
    id: 'market-target',
    source: 'market_growth',
    target: 'target',
    style: { stroke: '#222', strokeWidth: 2, strokeDasharray: '5,5' } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
];

        // Simplified DAG
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
    data: { label: 'Y (Sales)' },
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
    style: { stroke: '#222', strokeWidth: 3 } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
  {
    id: 'x2-y-simple',
    source: 'x2_simple',
    target: 'y_simple',
    style: { stroke: '#222', strokeWidth: 3 } as CSSProperties,
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
  },
];

let nodeId = 1000; // For generating new node IDs

// 判断节点是否为confounder
const isConfounder = (label: string) => {
  const l = label.toLowerCase();
  return (
    l.includes('confound') ||
    l.includes('christmas') ||
    l.includes('competitor') ||
    l.includes('market growth') ||
    l.includes('holiday')
  );
};

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
    (params: Edge | Connection) => {
      // 只处理source/target都为string的情况
      if (!params.source || !params.target) return;
      const sourceNode = nodes.find(n => n.id === params.source);
      let edgeStyle: CSSProperties = { stroke: '#222', strokeWidth: 2 };
      if (sourceNode) {
        const label = sourceNode.data.label || '';
        if (isConfounder(label)) {
          edgeStyle = { stroke: '#222', strokeWidth: 2, strokeDasharray: '5,5' };
        }
      }
      const newEdge: Edge = {
        ...params,
        id: `edge_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        style: edgeStyle,
        animated: false,
        source: params.source as string,
        target: params.target as string,
        markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
      };
      setEdges((eds) => addEdge(newEdge, eds));
    },
    [setEdges, nodes]
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

  // Export DAG structure
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

  // Generate Graphviz DOT format DAG string
  const generateDOTString = (): string => {
    let dotString = "digraph {\n";
    
    // Add edges (relationships)
    edges.forEach(edge => {
      dotString += `  ${edge.source} -> ${edge.target};\n`;
    });
    
    dotString += "}";
    return dotString;
  };

  // Train model
  const trainModel = async () => {
    if (nodes.length === 0) {
      alert('Please create DAG structure first!');
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
      console.log('Training result:', result); // Add debug log
      setTrainingResult(result);

    } catch (error) {
      console.error('Error training model:', error);
      
      let errorMessage = 'Unknown error';
      let errorDetails = '';
      
      if (error instanceof TypeError && error.message.includes('fetch')) {
        errorMessage = 'Cannot connect to backend server';
        errorDetails = 'Please ensure API server is running (python api_server.py)';
      } else if (error instanceof Error) {
        errorMessage = error.message;
        errorDetails = error.stack || '';
      }
      
      setTrainingResult({
        status: 'error',
        message: `Training failed: ${errorMessage}`,
        errorDetails: errorDetails,
        fullError: JSON.stringify(error, Object.getOwnPropertyNames(error))
      });
    } finally {
      setIsTraining(false);
    }
  };

  // Drag and drop to add nodes
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
        data: { label: `New ${template.label}` },
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

  // Select nodes
  const onNodeClick: NodeMouseHandler = useCallback((event, node) => {
    event.stopPropagation();
    setSelectedNodes([node.id]);
    setSelectedEdges([]);
  }, []);

  // Select edges
  const onEdgeClick: EdgeMouseHandler = useCallback((event, edge) => {
    event.stopPropagation();
    setSelectedEdges([edge.id]);
    setSelectedNodes([]);
  }, []);

  // Clear selection
  const onPaneClick = useCallback(() => {
    setSelectedNodes([]);
    setSelectedEdges([]);
  }, []);

  // Delete selected elements
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

  // Keyboard event handling
  const onKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.key === 'Delete' || event.key === 'Backspace') {
      deleteSelected();
    }
  }, [deleteSelected]);

  // Add keyboard listener
  React.useEffect(() => {
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [onKeyDown]);

  // Clear canvas
  const clearCanvas = () => {
    setNodes([]);
    setEdges([]);
    setSelectedNodes([]);
    setSelectedEdges([]);
    setTrainingResult(null);
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* Sidebar */}
      <div style={{
        width: '300px',
        background: 'white',
        borderRight: '1px solid #ddd',
        padding: '20px',
        overflow: 'auto',
        boxShadow: '2px 0 4px rgba(0,0,0,0.1)'
      }}>
        <h3 style={{ margin: '0 0 20px 0', color: '#333', fontSize: '18px' }}>Causal DAG Editor</h3>
        

        {/* Node toolbox */}
        <div style={{ marginBottom: '25px' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>Node Toolbox</h4>
          <p style={{ fontSize: '11px', color: '#888', margin: '0 0 15px 0' }}>
            Drag nodes below to canvas
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

        {/* Operation buttons */}
        <div style={{ marginBottom: '25px' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>Operations</h4>
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
              Delete Selected (Delete Key)
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
              Clear Canvas
            </button>
          </div>
        </div>

        {/* DAG export and model training */}
        <div style={{ marginBottom: '25px' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>Model Training</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
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
              {isTraining ? 'Training...' : '🚀 Train Causal Model'}
            </button>
          </div>

          {/* DAG export display */}
          {showDAGExport && (
            <div style={{ marginTop: '10px' }}>
              <h5 style={{ margin: '0 0 5px 0', fontSize: '12px', color: '#666' }}>DOT Format:</h5>
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

          {/* Training result display */}
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
                Training Result
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
                  <strong>Detailed Error Information:</strong>
                  {trainingResult.error_details && (
                    <div style={{ marginTop: '5px' }}>
                      <div><strong>Error Type:</strong> {trainingResult.error_details.error_type}</div>
                      <div><strong>Error Message:</strong> {trainingResult.error_details.error_message}</div>
                      {trainingResult.error_details.dag_type && (
                        <div><strong>DAG Type:</strong> {trainingResult.error_details.dag_type}</div>
                      )}
                      {trainingResult.error_details.nodes_count !== undefined && (
                        <div><strong>Node Count:</strong> {trainingResult.error_details.nodes_count}</div>
                      )}
                      {trainingResult.error_details.edges_count !== undefined && (
                        <div><strong>Edge Count:</strong> {trainingResult.error_details.edges_count}</div>
                      )}
                      <div><strong>Causal MMM Available:</strong> {trainingResult.error_details.causal_mmm_available ? 'Yes' : 'No'}</div>
                    </div>
                  )}
                  <details style={{ marginTop: '8px' }}>
                    <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>Technical Details (Click to expand)</summary>
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
                  <strong>Model Information:</strong>
                  <div style={{ margin: '5px 0', fontSize: '10px' }}>
                    <div><strong>DAG Type:</strong> {trainingResult.model_summary.dag_type}</div>
                    
                    {/* Model Type Indicator */}
                    {trainingResult.model_summary.model_type && (
                      <div style={{ marginTop: '4px' }}>
                        <strong>Model Type:</strong> 
                        <span style={{ 
                          marginLeft: '4px',
                          padding: '2px 6px',
                          borderRadius: '10px',
                          fontSize: '8px',
                          fontWeight: 'bold',
                          backgroundColor: trainingResult.model_summary.model_type === 'causal' ? '#4caf50' : '#ff9800',
                          color: 'white'
                        }}>
                          {trainingResult.model_summary.model_type === 'causal' ? '🧠 CAUSAL' : '📊 CORRELATIONAL'}
                        </span>
                      </div>
                    )}
                    
                    {/* Causal Constraints Indicator */}
                    {trainingResult.model_summary.has_causal_constraints !== undefined && (
                      <div style={{ marginTop: '2px', fontSize: '9px', color: '#666' }}>
                        <strong>Causal Constraints:</strong> 
                        <span style={{ 
                          marginLeft: '4px',
                          color: trainingResult.model_summary.has_causal_constraints ? '#2e7d32' : '#d32f2f'
                        }}>
                          {trainingResult.model_summary.has_causal_constraints ? '✅ Active' : '❌ None'}
                        </span>
                        {!trainingResult.model_summary.has_causal_constraints && (
                          <div style={{ fontSize: '8px', color: '#f57c00', marginTop: '2px' }}>
                            💡 Add control variables (like Holiday, Competitor) to enable causal constraints
                          </div>
                        )}
                      </div>
                    )}
                    
                    <div style={{ marginTop: '6px' }}>
                      <div><strong>Node Count:</strong> {trainingResult.model_summary.nodes_count}</div>
                      <div><strong>Edge Count:</strong> {trainingResult.model_summary.edges_count}</div>
                    </div>
                    
                    {trainingResult.model_summary.treatment_variables && (
                      <div><strong>Treatment Variables:</strong> {trainingResult.model_summary.treatment_variables.join(', ')}</div>
                    )}
                    {trainingResult.model_summary.outcome_variables && (
                      <div><strong>Outcome Variables:</strong> {trainingResult.model_summary.outcome_variables.join(', ')}</div>
                    )}
                    {trainingResult.model_summary.control_variables && trainingResult.model_summary.control_variables.length > 0 && (
                      <div><strong>Control Variables:</strong> {trainingResult.model_summary.control_variables.join(', ')}</div>
                    )}
                  </div>
                  
                  {trainingResult.model_summary.fit_quality && (
                    <div style={{ 
                      marginTop: '8px', 
                      padding: '8px', 
                      backgroundColor: '#e8f5e8', 
                      borderRadius: '4px',
                      border: '1px solid #c8e6c9'
                    }}>
                      <strong style={{ color: '#2e7d32' }}>🎯 Model Fit Quality:</strong>
                      <div style={{ marginTop: '4px', fontSize: '9px' }}>
                        <div><strong>R² Score:</strong> {trainingResult.model_summary.fit_quality.r2_score?.toFixed(4)} 
                          <span style={{ color: '#666', marginLeft: '5px' }}>
                            ({trainingResult.model_summary.fit_quality.r2_score && trainingResult.model_summary.fit_quality.r2_score > 0.8 ? 'Excellent' : 
                              trainingResult.model_summary.fit_quality.r2_score && trainingResult.model_summary.fit_quality.r2_score > 0.6 ? 'Good' : 'Fair'})
                          </span>
                        </div>
                        <div><strong>MAPE:</strong> {(trainingResult.model_summary.fit_quality.mape && (trainingResult.model_summary.fit_quality.mape * 100).toFixed(2))}%
                          <span style={{ color: '#666', marginLeft: '5px' }}>
                            ({trainingResult.model_summary.fit_quality.mape && trainingResult.model_summary.fit_quality.mape < 0.1 ? 'Excellent' : 
                              trainingResult.model_summary.fit_quality.mape && trainingResult.model_summary.fit_quality.mape < 0.2 ? 'Good' : 'Fair'})
                          </span>
                        </div>
                        <div><strong>MAE:</strong> {trainingResult.model_summary.fit_quality.mae?.toFixed(2)}</div>
                        <div><strong>RMSE:</strong> {trainingResult.model_summary.fit_quality.rmse?.toFixed(2)}</div>
                        <div><strong>Sample Size:</strong> {trainingResult.model_summary.fit_quality.sample_size}</div>
                      </div>
                    </div>
                  )}
                  
                  {trainingResult.model_summary.data_info && (
                    <div style={{ 
                      marginTop: '6px', 
                      padding: '6px', 
                      backgroundColor: '#f3e5f5', 
                      borderRadius: '4px',
                      border: '1px solid #e1bee7'
                    }}>
                      <strong style={{ color: '#7b1fa2' }}>📊 Data Statistics:</strong>
                      <div style={{ marginTop: '3px', fontSize: '9px' }}>
                        <div><strong>Actual Values:</strong> Mean={trainingResult.model_summary.data_info.actual_mean?.toFixed(2)}, Std={trainingResult.model_summary.data_info.actual_std?.toFixed(2)}</div>
                        <div><strong>Predicted Values:</strong> Mean={trainingResult.model_summary.data_info.prediction_mean?.toFixed(2)}, Std={trainingResult.model_summary.data_info.prediction_std?.toFixed(2)}</div>
                      </div>
                    </div>
                  )}
                  
                  {/* Dynamic Charts */}
                  {trainingResult.model_summary?.chart_data ? (
                    <div style={{ marginTop: '10px' }}>
                      <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#333' }}>📊 Model Evaluation Charts</h4>
                      
                      {/* Time Series Chart */}
                      {trainingResult.model_summary?.chart_data?.time_series && (
                        <div style={{ marginBottom: '20px', padding: '10px', backgroundColor: '#f8f9fa', borderRadius: '8px' }}>
                          <h5 style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#666' }}>
                            Time Series Fit (R² = {trainingResult.model_summary.fit_quality?.r2_score?.toFixed(3)})
                          </h5>
                          <div style={{ height: '300px' }}>
                            <Line
                                                             data={{
                                 labels: trainingResult.model_summary?.chart_data?.time_series?.dates || [],
                                 datasets: [
                                   {
                                     label: 'Observed',
                                     data: trainingResult.model_summary?.chart_data?.time_series?.actual || [],
                                    borderColor: '#000000',
                                    backgroundColor: 'transparent',
                                    borderWidth: 2,
                                    pointRadius: 0,
                                    tension: 0.1
                                  },
                                                                     {
                                     label: 'Predicted',
                                     data: trainingResult.model_summary?.chart_data?.time_series?.predicted || [],
                                    borderColor: '#1976d2',
                                    backgroundColor: 'transparent',
                                    borderWidth: 1.5,
                                    pointRadius: 0,
                                    tension: 0.1
                                  },
                                                                     {
                                     label: '95% HDI',
                                     data: trainingResult.model_summary?.chart_data?.time_series?.confidence_upper || [],
                                    borderColor: 'rgba(25, 118, 210, 0.2)',
                                    backgroundColor: 'rgba(25, 118, 210, 0.1)',
                                    borderWidth: 0,
                                    fill: '+1',
                                    pointRadius: 0,
                                    tension: 0.1
                                  },
                                                                     {
                                     label: '',
                                     data: trainingResult.model_summary?.chart_data?.time_series?.confidence_lower || [],
                                    borderColor: 'rgba(25, 118, 210, 0.2)',
                                    backgroundColor: 'rgba(25, 118, 210, 0.1)',
                                    borderWidth: 0,
                                    fill: false,
                                    pointRadius: 0,
                                    tension: 0.1,
                                    pointHoverRadius: 0,
                                    pointHitRadius: 0
                                  }
                                ]
                              }}
                              options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                  title: {
                                    display: true,
                                    text: 'Estimated Target Variable Over Time',
                                    font: { size: 14, weight: 'bold' }
                                  },
                                  legend: {
                                    display: true,
                                    position: 'top',
                                    labels: {
                                      filter: (legendItem) => legendItem.text !== ''
                                    }
                                  }
                                },
                                scales: {
                                  x: {
                                    title: {
                                      display: true,
                                      text: 'Days'
                                    }
                                  },
                                  y: {
                                    title: {
                                      display: true,
                                      text: 'Target Variable'
                                    }
                                  }
                                }
                              }}
                            />
                          </div>
                        </div>
                      )}
                      
                      {/* Media Contribution Recovery Chart */}
                      {trainingResult.model_summary?.chart_data?.media_contribution && (
                        <div style={{ marginBottom: '20px', padding: '10px', backgroundColor: '#f8f9fa', borderRadius: '8px' }}>
                          <h5 style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#666' }}>Media Contribution Recovery</h5>
                          
                          {/* X1 (Social Media) Chart */}
                          <div style={{ height: '300px', marginBottom: '20px' }}>
                            <Line
                              data={{
                                labels: trainingResult.model_summary.chart_data.media_contribution.dates,
                                datasets: [
                                  {
                                    label: 'Mean Recover x1 Effect',
                                    data: trainingResult.model_summary.chart_data.media_contribution.x1.predicted_mean,
                                    borderColor: '#1976d2',
                                    backgroundColor: 'transparent',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                    pointRadius: 0,
                                    tension: 0.1
                                  },
                                  {
                                    label: '95% Credible Interval',
                                    data: trainingResult.model_summary.chart_data.media_contribution.x1.confidence_upper,
                                    borderColor: 'rgba(25, 118, 210, 0.2)',
                                    backgroundColor: 'rgba(25, 118, 210, 0.2)',
                                    borderWidth: 0,
                                    fill: '+1',
                                    pointRadius: 0,
                                    tension: 0.1
                                  },
                                  {
                                    label: '',
                                    data: trainingResult.model_summary.chart_data.media_contribution.x1.confidence_lower,
                                    borderColor: 'rgba(25, 118, 210, 0.2)',
                                    backgroundColor: 'rgba(25, 118, 210, 0.2)',
                                    borderWidth: 0,
                                    fill: false,
                                    pointRadius: 0,
                                    tension: 0.1,
                                    pointHoverRadius: 0,
                                    pointHitRadius: 0
                                  },
                                  {
                                    label: 'Real x1 Effect',
                                    data: trainingResult.model_summary.chart_data.media_contribution.x1.real_effect,
                                    borderColor: '#000000',
                                    backgroundColor: 'transparent',
                                    borderWidth: 2,
                                    pointRadius: 0,
                                    tension: 0.1
                                  }
                                ]
                              }}
                              options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                  title: {
                                    display: true,
                                    text: 'X1 (Social Media) Contribution Recovery',
                                    font: { size: 12, weight: 'bold' }
                                  },
                                  legend: {
                                    display: true,
                                    position: 'top',
                                    labels: {
                                      filter: (legendItem) => legendItem.text !== ''
                                    }
                                  }
                                },
                                scales: {
                                  x: {
                                    title: {
                                      display: true,
                                      text: 'Date'
                                    }
                                  },
                                  y: {
                                    title: {
                                      display: true,
                                      text: 'Effect'
                                    }
                                  }
                                }
                              }}
                            />
                          </div>
                          
                          {/* X2 (Search Engine) Chart */}
                          <div style={{ height: '300px' }}>
                            <Line
                              data={{
                                labels: trainingResult.model_summary.chart_data.media_contribution.dates,
                                datasets: [
                                  {
                                    label: 'Mean Recover x2 Effect',
                                    data: trainingResult.model_summary.chart_data.media_contribution.x2.predicted_mean,
                                    borderColor: '#ff9800',
                                    backgroundColor: 'transparent',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                    pointRadius: 0,
                                    tension: 0.1
                                  },
                                  {
                                    label: '95% Credible Interval',
                                    data: trainingResult.model_summary.chart_data.media_contribution.x2.confidence_upper,
                                    borderColor: 'rgba(255, 152, 0, 0.2)',
                                    backgroundColor: 'rgba(255, 152, 0, 0.2)',
                                    borderWidth: 0,
                                    fill: '+1',
                                    pointRadius: 0,
                                    tension: 0.1
                                  },
                                  {
                                    label: '',
                                    data: trainingResult.model_summary.chart_data.media_contribution.x2.confidence_lower,
                                    borderColor: 'rgba(255, 152, 0, 0.2)',
                                    backgroundColor: 'rgba(255, 152, 0, 0.2)',
                                    borderWidth: 0,
                                    fill: false,
                                    pointRadius: 0,
                                    tension: 0.1,
                                    pointHoverRadius: 0,
                                    pointHitRadius: 0
                                  },
                                  {
                                    label: 'Real x2 Effect',
                                    data: trainingResult.model_summary.chart_data.media_contribution.x2.real_effect,
                                    borderColor: '#000000',
                                    backgroundColor: 'transparent',
                                    borderWidth: 2,
                                    pointRadius: 0,
                                    tension: 0.1
                                  }
                                ]
                              }}
                              options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                  title: {
                                    display: true,
                                    text: 'X2 (Search Engine) Contribution Recovery',
                                    font: { size: 12, weight: 'bold' }
                                  },
                                  legend: {
                                    display: true,
                                    position: 'top',
                                    labels: {
                                      filter: (legendItem) => legendItem.text !== ''
                                    }
                                  }
                                },
                                scales: {
                                  x: {
                                    title: {
                                      display: true,
                                      text: 'Date'
                                    }
                                  },
                                  y: {
                                    title: {
                                      display: true,
                                      text: 'Effect'
                                    }
                                  }
                                }
                              }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  ) : trainingResult.model_summary?.plot_available ? (
                    <div style={{ 
                      marginTop: '6px', 
                      padding: '6px', 
                      backgroundColor: '#fff3e0', 
                      borderRadius: '4px',
                      border: '1px solid #ffb74d'
                    }}>
                      <strong style={{ color: '#f57c00' }}>📊 Model evaluation completed</strong>
                      <div style={{ fontSize: '9px', color: '#666', marginTop: '2px' }}>
                        Chart data processing...
                      </div>
                    </div>
                  ) : null}
                  
                  <details style={{ marginTop: '8px' }}>
                    <summary style={{ cursor: 'pointer', fontWeight: 'bold', fontSize: '9px' }}>Technical Details (Click to expand)</summary>
                    <pre style={{ 
                      fontSize: '8px', 
                      margin: '5px 0 0 0', 
                      whiteSpace: 'pre-wrap',
                      maxHeight: '100px',
                      overflow: 'auto',
                      background: 'white',
                      padding: '5px',
                      borderRadius: '3px'
                    }}>
                      {JSON.stringify(trainingResult.model_summary, null, 2)}
                    </pre>
                  </details>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Documentation */}
        <div style={{ fontSize: '11px', color: '#666', lineHeight: '1.5' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>Instructions</h4>
          <ul style={{ margin: 0, paddingLeft: '15px' }}>
            <li>Drag nodes to canvas to add new nodes</li>
            <li>Drag node edge dots to create connections</li>
            <li>Click to select nodes or edges</li>
            <li>Press Delete key to delete selected elements</li>
            <li>Drag to move node positions</li>
            <li>Click Train Model after designing DAG</li>
          </ul>
        </div>
      </div>

      {/* Main Canvas Area */}
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
          defaultEdgeOptions={{
            type: 'default',
            markerEnd: 'arrowclosed',
          }}
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
