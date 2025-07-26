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

// Global ResizeObserver error suppression - must be at the top
const suppressResizeObserverErrors = () => {
  // Suppress console errors
  const originalConsoleError = console.error;
  console.error = (...args: any[]) => {
    if (args[0] && args[0].toString().includes('ResizeObserver loop')) {
      return; // Suppress ResizeObserver errors
    }
    originalConsoleError.apply(console, args);
  };

  // Suppress window errors
  const originalWindowError = window.onerror;
  window.onerror = (message, source, lineno, colno, error) => {
    if (typeof message === 'string' && message.includes('ResizeObserver loop')) {
      return true; // Suppress the error
    }
    return originalWindowError ? originalWindowError(message, source, lineno, colno, error) : false;
  };
};

// Call immediately when module loads
suppressResizeObserverErrors();

function App() {
  // Comprehensive ResizeObserver error suppression
  React.useLayoutEffect(() => {
    // Multiple approaches to catch ResizeObserver errors
    const originalError = console.error;
    console.error = (...args) => {
      if (args[0]?.toString().includes('ResizeObserver loop')) {
        return; // Suppress ResizeObserver errors
      }
      originalError.apply(console, args);
    };

    // Handle window errors
    const handleError = (event: ErrorEvent) => {
      if (event.error?.message?.includes('ResizeObserver loop') || 
          event.message?.includes('ResizeObserver loop')) {
        event.preventDefault();
        event.stopPropagation();
        return false;
      }
    };

    // Handle unhandled promise rejections
    const handleRejection = (event: PromiseRejectionEvent) => {
      if (event.reason?.message?.includes('ResizeObserver loop')) {
        event.preventDefault();
        return false;
      }
    };

    // Set up error handlers
    window.addEventListener('error', handleError, true);
    window.addEventListener('unhandledrejection', handleRejection, true);
    
    // Override the global error handler
    const originalOnError = window.onerror;
    window.onerror = (message, source, lineno, colno, error) => {
      if (typeof message === 'string' && message.includes('ResizeObserver loop')) {
        return true; // Suppress the error
      }
      return originalOnError ? originalOnError(message, source, lineno, colno, error) : false;
    };

    return () => {
      console.error = originalError;
      window.removeEventListener('error', handleError, true);
      window.removeEventListener('unhandledrejection', handleRejection, true);
      window.onerror = originalOnError;
    };
  }, []);

  const [dagType, setDagType] = useState<'simple' | 'realistic' | 'complex'>('realistic');
  const [nodes, setNodes, onNodesChange] = useNodesState(businessScenarioNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(businessScenarioEdges);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [selectedEdges, setSelectedEdges] = useState<string[]>([]);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);
  const [showDAGExport, setShowDAGExport] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [useCustomData, setUseCustomData] = useState(false);
  const [fileSchema, setFileSchema] = useState<string[] | null>(null);
  const [isParsingFile, setIsParsingFile] = useState(false);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // Resizable panels state
  const [leftPanelWidth, setLeftPanelWidth] = useState(50); // percentage
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef({ isActive: false, lastUpdate: 0, timeoutId: null as NodeJS.Timeout | null });

  // Handle splitter drag
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    dragRef.current.isActive = true;
    e.preventDefault();
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!dragRef.current.isActive) return;
    
    // Clear any pending timeout
    if (dragRef.current.timeoutId) {
      clearTimeout(dragRef.current.timeoutId);
    }
    
    const containerWidth = window.innerWidth;
    const newLeftWidth = (e.clientX / containerWidth) * 100;
    
    // Constrain between 20% and 80%
    const constrainedWidth = Math.min(Math.max(newLeftWidth, 20), 80);
    
    // Debounce the state update
    dragRef.current.timeoutId = setTimeout(() => {
      setLeftPanelWidth(constrainedWidth);
    }, 10); // 10ms debounce
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    dragRef.current.isActive = false;
    
    // Clear any pending timeout
    if (dragRef.current.timeoutId) {
      clearTimeout(dragRef.current.timeoutId);
      dragRef.current.timeoutId = null;
    }
    
    // Trigger ReactFlow resize after drag ends
    setTimeout(() => {
      if (reactFlowInstance) {
        reactFlowInstance.fitView();
      }
    }, 200); // Increased delay to let everything settle
  }, [reactFlowInstance]);

  // Add global mouse event listeners
  React.useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

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

  const switchDAG = (type: 'simple' | 'realistic' | 'complex') => {
    setDagType(type);
    if (type === 'realistic') {
      setNodes(businessScenarioNodes);
      setEdges(businessScenarioEdges);
    } else if (type === 'simple') {
      setNodes(simpleDagNodes);
      setEdges(simpleDagEdges);
    } else if (type === 'complex') {
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

      let response;
      
      if (useCustomData && uploadedFile) {
        // Train with uploaded file
        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('dag_structure', JSON.stringify(dagExport));
        formData.append('dag_dot_string', dotString);
        formData.append('dag_type', dagType);

        response = await fetch('http://localhost:8000/train-model-with-file', {
          method: 'POST',
          body: formData,
        });
      } else {
        // Train with default data (data_mmm.csv)
        response = await fetch('http://localhost:8000/train-model', {
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
      }

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

  // File upload handlers
  const parseCSVHeaders = (file: File): Promise<string[]> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const text = e.target?.result as string;
          const firstLine = text.split('\n')[0];
          const headers = firstLine.split(',').map(header => header.trim().replace(/"/g, ''));
          resolve(headers);
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  };

  const handleFileUpload = async (file: File) => {
    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
      alert('Please upload a CSV file only.');
      return;
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      alert('File size should be less than 10MB.');
      return;
    }
    
    setIsParsingFile(true);
    try {
      const headers = await parseCSVHeaders(file);
      setFileSchema(headers);
      setUploadedFile(file);
      setUseCustomData(true);
      
      // Generate initial column interpretation
      const interpretation = analyzeColumnMeanings(headers);
      setColumnInterpretation(interpretation);
      setIsInterpretationConfirmed(false);
      setCausalAnalysis(null);
      
      // Ask user if they want to clear the current DAG to start fresh
      if (nodes.length > 0) {
        const shouldClear = window.confirm(
          'You have an existing DAG. Would you like to clear it and start fresh with your new data columns?'
        );
        if (shouldClear) {
          setNodes([]);
          setEdges([]);
        }
      }
    } catch (error) {
      console.error('Error parsing CSV:', error);
      alert('Failed to parse CSV file. Please check the file format.');
    } finally {
      setIsParsingFile(false);
    }
  };

  const handleFileDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleFileDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const removeUploadedFile = () => {
    setUploadedFile(null);
    setUseCustomData(false);
    setFileSchema(null);
    setColumnInterpretation(null);
    setIsInterpretationConfirmed(false);
    setCausalAnalysis(null);
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

      let template: NodeTemplate | undefined;
      
      if (fileSchema && fileSchema.includes(type)) {
        // Handle CSV column names
        const dynamicTemplates = createDynamicNodeTemplates(fileSchema);
        template = dynamicTemplates.find(t => t.type === type);
      } else {
        // Handle default node templates
        template = nodeTemplates.find(t => t.type === type);
      }
      
      if (!template) return;

      const newNode: Node = {
        id: `${type}_${nodeId++}`,
        type: 'default',
        position,
        data: { label: template.label },
        style: template.style,
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes, fileSchema]
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

  // Helper function to guess variable type based on column name
  const guessVariableType = (columnName: string): 'treatment' | 'outcome' | 'confounder' | 'unknown' => {
    const name = columnName.toLowerCase();
    
    // Treatment variables (marketing channels)
    if (name.includes('x1') || name.includes('x2') || 
        name.includes('social') || name.includes('search') || 
        name.includes('tv') || name.includes('radio') || 
        name.includes('display') || name.includes('video') ||
        name.includes('channel') || name.includes('campaign') ||
        name.includes('ad') || name.includes('marketing') ||
        name.includes('facebook') || name.includes('google') ||
        name.includes('instagram') || name.includes('tiktok')) {
      return 'treatment';
    }
    
    // Outcome variables
    if (name.includes('y') || name.includes('target') || 
        name.includes('sales') || name.includes('revenue') || 
        name.includes('conversion') || name.includes('purchase') ||
        name.includes('order') || name.includes('transaction')) {
      return 'outcome';
    }
    
    // Confounder variables (including new ones)
    if (name.includes('holiday') || name.includes('season') || 
        name.includes('competitor') || name.includes('event') || 
        name.includes('weather') || name.includes('trend') ||
        name.includes('macro') || name.includes('gdp') ||
        name.includes('index') || name.includes('economic') ||
        name.includes('inflation') || name.includes('activity') ||
        name === 'seasonality' || name === 'competitor_activity' ||
        name === 'economic_environment') {
      return 'confounder';
    }
    
    // Default to unknown for other variables
    return 'unknown';
  };

  // Create dynamic node templates from file schema
  const createDynamicNodeTemplates = (schema: string[]): NodeTemplate[] => {
    return schema.map(columnName => {
      const type = guessVariableType(columnName);
      
      return {
        type: columnName, // Use column name as type for uniqueness
        label: columnName,
        style: {
          backgroundColor: '#1976d2', // Strong blue for all user data
          color: 'white',
          padding: '10px 15px',
          borderRadius: '8px',
          border: '2px solid #0d47a1',
          fontSize: '12px',
          fontWeight: 'bold',
          cursor: 'grab',
          minWidth: '100px',
          textAlign: 'center' as const
        },
        description: `用户数据列: ${columnName} (${type})`
      };
    });
  };

  // Generate recommended DAG structures based on detected variables
  const generateRecommendedDAGs = (schema: string[], interpretation?: {[key: string]: {type: string, description: string, confidence: number}}) => {
    // Filter out time/date columns
    const nonTimeColumns = schema.filter(col => {
      const type = interpretation?.[col]?.type || guessVariableType(col);
      return type !== 'time' && 
        !col.toLowerCase().includes('date') && 
        !col.toLowerCase().includes('time') && 
        !col.toLowerCase().includes('week') && 
        !col.toLowerCase().includes('month');
    });
    
    // Identify variable types using confirmed interpretation or fallback to guess
    const getVariableType = (col: string) => interpretation?.[col]?.type || guessVariableType(col);
    
    const treatments = nonTimeColumns.filter(col => getVariableType(col) === 'treatment');
    const outcomes = nonTimeColumns.filter(col => getVariableType(col) === 'outcome');
    const confounders = nonTimeColumns.filter(col => getVariableType(col) === 'confounder');
    const mediators = nonTimeColumns.filter(col => getVariableType(col) === 'mediator');
    
    // Add common confounders if not present in data
    const commonConfounders = ['seasonality', 'competitor_activity', 'economic_environment'];
    const theoreticalConfounders: string[] = [];
    
    // Add missing common confounders for demonstration
    commonConfounders.forEach(conf => {
      if (!confounders.some(existing => 
        existing.toLowerCase().includes(conf.replace('_', '').replace('activity', '').replace('environment', '')))) {
        theoreticalConfounders.push(conf);
      }
    });
    
    // Use ALL user columns, not just primary ones
    const allTreatments = treatments.length > 0 ? treatments : ['x1', 'x2'];
    const allOutcomes = outcomes.length > 0 ? outcomes : ['y'];
    const allConfounders = [...confounders, ...theoreticalConfounders.slice(0, 2)]; // Limit theoretical ones
    // Remove mediators - not needed for now

    return [
      {
        name: 'Simple Direct Effects',
        description: '直接因果关系，包含所有变量',
        nodes: [
          // Include ALL treatments
          ...allTreatments.map((treatment, idx) => ({
            id: treatment,
            label: treatment,
            position: { x: 150, y: 100 + (idx * 80) },
            isUserData: schema.includes(treatment)
          })),
          // Include ALL outcomes
          ...allOutcomes.map((outcome, idx) => ({
            id: outcome,
            label: outcome,
            position: { x: 400, y: 130 + (idx * 80) },
            isUserData: schema.includes(outcome)
          }))
        ],
        edges: [
          // Connect all treatments to all outcomes
          ...allTreatments.flatMap(treatment =>
            allOutcomes.map((outcome, idx) => ({
              id: `${treatment}_to_${outcome}`,
              source: treatment,
              target: outcome
            }))
          )
        ]
      },
      {
        name: 'With Confounders',
        description: '包含所有变量和混淆因子',
        nodes: [
          // All treatments
          ...allTreatments.map((treatment, idx) => ({
            id: treatment,
            label: treatment,
            position: { x: 250, y: 120 + (idx * 80) },
            isUserData: schema.includes(treatment)
          })),
          // All outcomes
          ...allOutcomes.map((outcome, idx) => ({
            id: outcome,
            label: outcome,
            position: { x: 450, y: 140 + (idx * 80) },
            isUserData: schema.includes(outcome)
          })),
          // All confounders (user + theoretical)
          ...allConfounders.slice(0, 3).map((confounder, idx) => ({
            id: confounder,
            label: confounder,
            position: { x: 80, y: 80 + (idx * 60) },
            isUserData: schema.includes(confounder)
          }))
        ],
        edges: [
          // Treatments to outcomes
          ...allTreatments.flatMap(treatment =>
            allOutcomes.map(outcome => ({
              id: `${treatment}_to_${outcome}`,
              source: treatment,
              target: outcome
            }))
          ),
          // Confounders to treatments and outcomes
          ...allConfounders.slice(0, 3).flatMap(confounder => [
            ...allTreatments.map(treatment => ({
              id: `${confounder}_to_${treatment}`,
              source: confounder,
              target: treatment
            })),
            ...allOutcomes.map(outcome => ({
              id: `${confounder}_to_${outcome}`,
              source: confounder,
              target: outcome
            }))
          ])
        ]
      }
    ];
  };

  // Apply recommended DAG to canvas
  const applyRecommendedDAG = (dagStructure: any) => {
    // Create nodes with proper styling
    const styledNodes = dagStructure.nodes.map((node: any) => {
      const varType = guessVariableType(node.label);
      const baseTemplate = nodeTemplates.find(t => t.type === varType) || nodeTemplates[0];
      
      // Check if node is in user's actual data
      const isInUserData = fileSchema ? fileSchema.includes(node.id) : true;
      
      // Color scheme: User data = blue, Theoretical = different colors by type
      let backgroundColor;
      if (isInUserData) {
        // All user-provided columns use the same blue color
        backgroundColor = '#1976d2'; // Strong blue for user data
      } else {
        // Theoretical variables use different colors by type
        const colorMap = {
          'treatment': '#42a5f5',      // Light blue
          'outcome': '#ef5350',        // Red
          'confounder': '#ff9800',     // Orange
          'seasonality': '#ffb74d',    // Light orange
          'competitor': '#8d6e63',     // Brown
          'economic_activity': '#78909c', // Blue grey
          'unknown': '#9e9e9e'         // Grey
        };
        backgroundColor = colorMap[varType] || '#9e9e9e'; // Default grey
      }
      
      // Check if this node is a confounder
      const nodeType = columnInterpretation && columnInterpretation[node.label] 
        ? columnInterpretation[node.label].type 
        : guessVariableType(node.label);
      const isConfounder = nodeType === 'confounder';
      
      // Create different styles for user data vs theoretical variables
      const nodeStyle = {
        ...baseTemplate.style,
        backgroundColor,
        color: 'white', // White text for better contrast
        // User data nodes: larger and more prominent
        // Theoretical nodes: smaller and less prominent
        padding: isInUserData ? '15px 20px' : '8px 12px',
        fontSize: isInUserData ? '14px' : '11px',
        fontWeight: isInUserData ? 'bold' : 'normal',
        borderWidth: isInUserData ? '3px' : '2px',
        borderColor: isInUserData ? '#0d47a1' : backgroundColor,
        borderStyle: isConfounder ? 'dashed' : 'solid', // Dashed border for confounders
        minWidth: isInUserData ? '120px' : '100px',
        minHeight: isInUserData ? '50px' : '35px'
      };
      
      return {
        id: node.id,
        type: 'default',
        position: node.position,
        data: { label: node.label },
        style: nodeStyle,
      };
    });

    // Create edges with proper styling
    const styledEdges = dagStructure.edges.map((edge: any) => {
      // Check if BOTH source AND target are in user's data
      const isSourceInUserData = fileSchema ? fileSchema.includes(edge.source) : true;
      const isTargetInUserData = fileSchema ? fileSchema.includes(edge.target) : true;
      
      // Check if source or target is a confounder
      const getNodeType = (nodeId: string, nodeLabel?: string) => {
        const id = nodeLabel || nodeId;
        if (columnInterpretation && columnInterpretation[id]) {
          return columnInterpretation[id].type;
        }
        return guessVariableType(id);
      };
      
      const isSourceConfounder = getNodeType(edge.source, edge.source) === 'confounder';
      const isTargetConfounder = getNodeType(edge.target, edge.target) === 'confounder';
      
      // Use dashed line ONLY if EITHER source OR target is a confounder
      // Mediators and other theoretical variables should use solid lines for causal pathways
      const shouldUseDashedLine = isSourceConfounder || isTargetConfounder;
      
      // Make user data connections more prominent
      const strokeWidth = isSourceInUserData && isTargetInUserData ? 3 : 2;
      
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        style: {
          ...edge.style,
          stroke: selectedEdges.includes(edge.id) ? '#2196f3' : edge.style?.stroke,
          strokeWidth: selectedEdges.includes(edge.id) ? 4 : strokeWidth,
          strokeDasharray: shouldUseDashedLine ? '8,4' : 'none'
        },
        animated: false,
        markerEnd: { type: MarkerType.ArrowClosed, color: '#222' },
      };
    });

    setNodes(styledNodes);
    setEdges(styledEdges);
    setSelectedNodes([]);
    setSelectedEdges([]);
  };

  // Add new state for column interpretation workflow
  const [columnInterpretation, setColumnInterpretation] = useState<{[key: string]: {type: string, description: string, confidence: number}} | null>(null);
  const [isInterpretationConfirmed, setIsInterpretationConfirmed] = useState(false);
  const [causalAnalysis, setCausalAnalysis] = useState<string | null>(null);

  // Analyze column meanings and generate interpretation
  const analyzeColumnMeanings = (schema: string[]) => {
    const interpretation: {[key: string]: {type: string, description: string, confidence: number}} = {};
    
    schema.forEach(col => {
      const colLower = col.toLowerCase();
      let type = 'unknown';
      let description = '';
      let confidence = 0.5; // Default confidence
      
      // Date/Time columns
      if (colLower.includes('date') || colLower.includes('time') || colLower.includes('week') || colLower.includes('month')) {
        type = 'time';
        description = '时间变量（通常用于时间序列分析）';
        confidence = 0.9;
      }
      // Outcome variables
      else if (colLower.includes('sales') || colLower.includes('revenue') || colLower.includes('conversion') || 
               colLower.includes('purchase') || colLower.includes('profit') || colLower === 'y') {
        type = 'outcome';
        description = '结果变量（我们想要预测/解释的目标）';
        confidence = 0.8;
      }
      // Marketing/Treatment variables
      else if (colLower.includes('ad') || colLower.includes('marketing') || colLower.includes('campaign') ||
               colLower.includes('social') || colLower.includes('search') || colLower.includes('tv') ||
               colLower.includes('radio') || colLower.includes('email') || colLower.match(/x\d+/)) {
        type = 'treatment';
        description = '营销/干预变量（广告投入、营销活动等）';
        confidence = 0.8;
      }
      // Holiday/Event variables
      else if (colLower.includes('holiday') || colLower.includes('event') || colLower.includes('promotion')) {
        type = 'confounder';
        description = '外部事件/促销变量（可能影响营销效果和结果）';
        confidence = 0.7;
      }
      // Price variables
      else if (colLower.includes('price') || colLower.includes('cost') || colLower.includes('discount')) {
        type = 'confounder';
        description = '定价相关变量（可能同时影响营销策略和销售）';
        confidence = 0.7;
      }
      // Engagement/Mediator variables
      else if (colLower.includes('click') || colLower.includes('impression') || colLower.includes('engagement') ||
               colLower.includes('traffic') || colLower.includes('awareness')) {
        type = 'mediator';
        description = '中介变量（营销影响的中间过程）';
        confidence = 0.7;
      }
      // Competition variables
      else if (colLower.includes('competitor') || colLower.includes('market_share')) {
        type = 'confounder';
        description = '竞争相关变量（外部市场因素）';
        confidence = 0.7;
      }
      // Economic variables
      else if (colLower.includes('gdp') || colLower.includes('economic') || colLower.includes('unemployment')) {
        type = 'confounder';
        description = '经济环境变量（宏观经济因素）';
        confidence = 0.8;
      }
      else {
        // Try to guess based on context
        if (colLower.includes('budget') || colLower.includes('spend') || colLower.includes('investment')) {
          type = 'treatment';
          description = '投入/预算变量（可能是营销投入）';
          confidence = 0.6;
        } else if (colLower.includes('customer') || colLower.includes('user')) {
          type = 'outcome';
          description = '客户相关指标（可能是结果变量）';
          confidence = 0.5;
        } else {
          type = 'unknown';
          description = '未知类型（请手动分类）';
          confidence = 0.3;
        }
      }
      
      interpretation[col] = { type, description, confidence };
    });
    
    return interpretation;
  };

  // Generate causal analysis based on confirmed interpretation
  const generateCausalAnalysis = (interpretation: {[key: string]: {type: string, description: string, confidence: number}}) => {
    const treatments = Object.keys(interpretation).filter(col => interpretation[col].type === 'treatment');
    const outcomes = Object.keys(interpretation).filter(col => interpretation[col].type === 'outcome');
    const confounders = Object.keys(interpretation).filter(col => interpretation[col].type === 'confounder');
    
    let analysis = "## 🔍 因果推断分析\n\n";
    
    // Main causal relationships
    analysis += "### 主要因果关系:\n";
    if (treatments.length > 0 && outcomes.length > 0) {
      treatments.forEach(treatment => {
        outcomes.forEach(outcome => {
          analysis += `• **${treatment}** → **${outcome}**: 营销投入对结果的直接因果效应\n`;
        });
      });
    }
    
    // Confounding analysis
    if (confounders.length > 0) {
      analysis += "\n### ⚠️ 混淆因子分析:\n";
      confounders.forEach(confounder => {
        analysis += `• **${confounder}**: 可能同时影响营销投入和结果，需要控制其影响\n`;
      });
      analysis += "\n**建议**: 在DAG中包含这些混淆因子，以获得准确的因果效应估计。\n";
    }
    
    // DAG recommendations
    analysis += "\n### 📊 DAG结构建议:\n";
    
    if (confounders.length === 0) {
      analysis += "1. **简单直接模型**: 由于没有明显的混淆因子，可以使用简单的直接因果关系\n";
      analysis += "2. **风险**: 可能存在未观察到的混淆，建议考虑理论上的混淆因子\n";
    } else {
      analysis += "1. **控制混淆模型**: 重点控制混淆因子，获得无偏的因果效应估计\n";
      analysis += "2. **推荐**: 使用包含混淆因子的DAG结构\n";
    }
    
    analysis += "\n### 🎯 最佳实践建议:\n";
    analysis += "• 从简单模型开始，逐步增加复杂性\n";
    analysis += "• 基于领域知识验证因果假设\n";
    analysis += "• 比较不同DAG结构的模型性能\n";
    analysis += "• 注意因果识别的假设条件\n";
    
    return analysis;
  };

  // Update column interpretation
  const updateColumnInterpretation = (column: string, newType: string) => {
    if (columnInterpretation) {
      setColumnInterpretation({
        ...columnInterpretation,
        [column]: {
          ...columnInterpretation[column],
          type: newType
        }
      });
    }
  };

  // Confirm interpretation and generate analysis
  const confirmInterpretation = () => {
    if (columnInterpretation) {
      setIsInterpretationConfirmed(true);
      const analysis = generateCausalAnalysis(columnInterpretation);
      setCausalAnalysis(analysis);
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* Sidebar */}
      <div style={{
        width: `${leftPanelWidth}%`, // Simple dynamic width
        background: 'white',
        borderRight: '1px solid #ddd',
        padding: '20px',
        overflowY: 'auto',
        fontSize: '13px'
      }}>
        <h3 style={{ margin: '0 0 20px 0', color: '#333', fontSize: '18px' }}>Causal DAG Editor</h3>
        


        {/* Recommended DAG structures - show only after causal analysis is confirmed */}
        {fileSchema && isInterpretationConfirmed && (
          <div style={{ marginBottom: '25px' }}>
            <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>
              🎯 基于因果分析的DAG推荐
            </h4>
            <p style={{ fontSize: '10px', color: '#666', margin: '0 0 15px 0' }}>
              根据您确认的变量类型和因果分析，选择最适合的DAG结构：
            </p>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '15px' }}>
              {generateRecommendedDAGs(fileSchema, columnInterpretation || undefined).map((dag, index) => (
                <div
                  key={index}
                  style={{
                    border: '1px solid #ddd',
                    borderRadius: '8px',
                    backgroundColor: '#fafafa',
                    overflow: 'hidden',
                    transition: 'all 0.2s ease',
                  }}
                >
                  {/* Header */}
                  <div style={{
                    padding: '12px',
                    backgroundColor: '#f5f5f5',
                    borderBottom: '1px solid #e0e0e0'
                  }}>
                    <div style={{ fontSize: '11px', fontWeight: 'bold', color: '#333', marginBottom: '4px' }}>
                      {index + 1}. {dag.name}
                    </div>
                    <div style={{ fontSize: '9px', color: '#666' }}>
                      {dag.description}
                    </div>
                    <div style={{ fontSize: '8px', color: '#888', marginTop: '4px' }}>
                      {dag.nodes.length} 节点 • {dag.edges.length} 关系
                    </div>
                  </div>
                  
                  {/* Visual Preview */}
                  <div style={{ 
                    padding: '15px',
                    backgroundColor: 'white',
                    position: 'relative',
                    height: '120px',
                    overflow: 'hidden'
                  }}>
                    <svg 
                      width="100%" 
                      height="100%" 
                      viewBox="0 0 300 100"
                      style={{ 
                        border: '1px solid #f0f0f0',
                        borderRadius: '4px',
                        backgroundColor: '#fefefe'
                      }}
                    >
                      {/* Draw edges first */}
                      {dag.edges.map((edge, edgeIndex) => {
                        const sourceNodeIndex = dag.nodes.findIndex(n => n.id === edge.source);
                        const targetNodeIndex = dag.nodes.findIndex(n => n.id === edge.target);
                        
                        if (sourceNodeIndex === -1 || targetNodeIndex === -1) return null;
                        
                        // Check if source or target node is not in user's actual data
                        const sourceNode = dag.nodes[sourceNodeIndex];
                        const targetNode = dag.nodes[targetNodeIndex];
                        const isSourceInUserData = fileSchema.includes(sourceNode.id);
                        const isTargetInUserData = fileSchema.includes(targetNode.id);
                        
                        // Check if source or target is a confounder
                        const getNodeType = (nodeId: string, nodeLabel?: string) => {
                          const id = nodeLabel || nodeId;
                          if (columnInterpretation && columnInterpretation[id]) {
                            return columnInterpretation[id].type;
                          }
                          return guessVariableType(id);
                        };
                        
                        const isSourceConfounder = getNodeType(sourceNode.id, sourceNode.label) === 'confounder';
                        const isTargetConfounder = getNodeType(targetNode.id, targetNode.label) === 'confounder';
                        
                        // Use dashed line ONLY if EITHER source OR target is a confounder
                        // Mediators and other theoretical variables should use solid lines for causal pathways
                        const shouldUseDashedLine = isSourceConfounder || isTargetConfounder;
                        
                        // Simple layout: distribute nodes horizontally
                        const nodeWidth = 280 / Math.max(dag.nodes.length, 3);
                        const sourceX = 20 + sourceNodeIndex * nodeWidth;
                        const targetX = 20 + targetNodeIndex * nodeWidth;
                        const y = 50;
                        
                        // Create curved path for better visualization
                        const midX = (sourceX + targetX) / 2;
                        const midY = sourceX < targetX ? y - 20 : y + 20;
                        
                        return (
                          <path
                            key={edgeIndex}
                            d={`M ${sourceX} ${y} Q ${midX} ${midY} ${targetX} ${y}`}
                            stroke="#666"
                            strokeWidth="1.5"
                            strokeDasharray={shouldUseDashedLine ? "4,3" : "none"}
                            fill="none"
                            markerEnd={`url(#arrow-${index})`}
                          />
                        );
                      })}
                      
                      {/* Draw nodes */}
                      {dag.nodes.map((node, nodeIndex) => {
                        const nodeWidth = 280 / Math.max(dag.nodes.length, 3);
                        const x = 20 + nodeIndex * nodeWidth;
                        const y = 50;
                        
                        // Check if node is in user's actual data
                        const isInUserData = fileSchema.includes(node.id);
                        
                        // Different sizes for user data vs theoretical variables
                        const nodeRadius = isInUserData ? 18 : 12;
                        const fontSize = isInUserData ? 9 : 7;
                        const fontWeight = isInUserData ? "bold" : "normal";
                        
                        // Color scheme: User data = blue, Theoretical = different colors by type
                        let nodeColor;
                        if (isInUserData) {
                          // All user-provided columns use the same blue color
                          nodeColor = '#1976d2'; // Strong blue for user data
                        } else {
                          // Theoretical variables use different colors by type
                          const varType = guessVariableType(node.label);
                          nodeColor = {
                            'treatment': '#42a5f5',      // Light blue
                            'outcome': '#ef5350',        // Red
                            'confounder': '#ff9800',     // Orange
                            'seasonality': '#ffb74d',    // Light orange
                            'competitor': '#8d6e63',     // Brown
                            'economic_activity': '#78909c', // Blue grey
                            'unknown': '#9e9e9e'         // Grey
                          }[varType] || '#9e9e9e';       // Default grey
                        }
                        
                        // Check if this node is a confounder
                        const nodeType = columnInterpretation && columnInterpretation[node.label] 
                          ? columnInterpretation[node.label].type 
                          : guessVariableType(node.label);
                        const isConfounder = nodeType === 'confounder';
                        
                        return (
                          <g key={nodeIndex}>
                            <circle
                              cx={x}
                              cy={y}
                              r={nodeRadius}
                              fill={nodeColor}
                              stroke="white"
                              strokeWidth={isInUserData ? "3" : "2"}
                              strokeDasharray={isConfounder ? "4,2" : "none"}
                            />
                            <text
                              x={x}
                              y={y + nodeRadius + 8}
                              textAnchor="middle"
                              fontSize={fontSize}
                              fill="#333"
                              fontWeight={fontWeight}
                            >
                              {node.label.length > 8 ? node.label.substring(0, 8) + '...' : node.label}
                            </text>
                            {/* Add indicator for theoretical variables */}
                            {!isInUserData && (
                              <text
                                x={x}
                                y={y + 3}
                                textAnchor="middle"
                                fontSize="8"
                                fill="white"
                                fontWeight="bold"
                              >
                                ?
                              </text>
                            )}
                          </g>
                        );
                      })}
                      
                      {/* Arrow marker definition */}
                      <defs>
                        <marker
                          id={`arrow-${index}`}
                          viewBox="0 0 10 10"
                          refX="9"
                          refY="3"
                          markerWidth="6"
                          markerHeight="6"
                          orient="auto"
                        >
                          <path d="M0,0 L0,6 L9,3 z" fill="#666"/>
                        </marker>
                      </defs>
                    </svg>
                  </div>
                  
                  {/* Apply Button */}
                  <div style={{ 
                    padding: '12px',
                    backgroundColor: '#f5f5f5',
                    borderTop: '1px solid #e0e0e0'
                  }}>
                    <button
                      onClick={() => applyRecommendedDAG(dag)}
                      style={{
                        width: '100%',
                        padding: '8px 16px',
                        backgroundColor: '#2196f3',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontWeight: 'bold',
                        cursor: 'pointer',
                        transition: 'background-color 0.2s'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#1976d2'}
                      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#2196f3'}
                    >
                      应用此结构
                    </button>
                  </div>
                </div>
              ))}
            </div>
            
            <div style={{ 
              fontSize: '9px', 
              color: '#666', 
              marginTop: '12px',
              padding: '8px',
              backgroundColor: '#fffbf0',
              border: '1px solid #ffd54f',
              borderRadius: '4px'
            }}>
              💡 <strong>提示：</strong> 应用任何结构后，你都可以通过添加/删除节点和连接来修改它。
            </div>
            
            {/* Legend for DAG visualization */}
            <div style={{ 
              fontSize: '9px', 
              color: '#666', 
              marginTop: '8px',
              padding: '8px',
              backgroundColor: '#f0f7ff',
              border: '1px solid #2196f3',
              borderRadius: '4px'
            }}>
              <strong>🔍 图例说明：</strong><br />
              <span style={{ fontSize: '8px' }}>
                • <strong>实线节点/边</strong>：你数据中实际存在的变量<br />
                • <strong>虚线节点/边</strong>：理论上的混淆变量（建议考虑但可能未测量）<br />
                • <strong>? 标记</strong>：表示该变量在你的数据中不存在，但对因果推断很重要
              </span>
            </div>
          </div>
        )}

        {/* Operation buttons */}
        <div style={{ marginBottom: '25px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          </div>
        </div>

        {/* DAG export and model training */}
        <div style={{ marginBottom: '25px' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>Data Source</h4>
          
          {/* Data source selection */}
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', fontSize: '12px' }}>
              <input
                type="radio"
                name="dataSource"
                checked={!useCustomData}
                onChange={() => {
                  setUseCustomData(false);
                  setUploadedFile(null);
                }}
              />
              Use default dataset (data_mmm.csv)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
              <input
                type="radio"
                name="dataSource"
                checked={useCustomData}
                onChange={() => setUseCustomData(true)}
              />
              Upload custom CSV file
            </label>
          </div>

          {/* File upload area */}
          {useCustomData && (
            <div style={{ marginBottom: '15px' }}>
              {!uploadedFile ? (
                <div>
                  <div
                    onDragOver={handleFileDragOver}
                    onDragLeave={handleFileDragLeave}
                    onDrop={handleFileDrop}
                    style={{
                      border: `2px dashed ${isDragOver ? '#2196f3' : '#ddd'}`,
                      borderRadius: '8px',
                      padding: '20px',
                      textAlign: 'center',
                      backgroundColor: isDragOver ? '#f3f9ff' : '#fafafa',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      marginBottom: '10px'
                    }}
                    onClick={() => document.getElementById('file-input')?.click()}
                  >
                    <div style={{ fontSize: '24px', marginBottom: '8px' }}>📁</div>
                    <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>
                      Drag & drop your CSV file here, or click to browse
                    </div>
                    <div style={{ fontSize: '10px', color: '#999' }}>
                      Max file size: 10MB
                    </div>
                  </div>
                  
                  <input
                    id="file-input"
                    type="file"
                    accept=".csv"
                    onChange={handleFileInputChange}
                    style={{ display: 'none' }}
                  />
                </div>
              ) : (
                <div style={{
                  padding: '12px',
                  backgroundColor: '#e8f5e9',
                  border: '1px solid #4caf50',
                  borderRadius: '4px',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontSize: '16px' }}>📄</span>
                      <div>
                        <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#2e7d32' }}>
                          {uploadedFile.name}
                        </div>
                        <div style={{ fontSize: '10px', color: '#666' }}>
                          {(uploadedFile.size / 1024).toFixed(1)} KB • {fileSchema?.length || 0} columns
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={removeUploadedFile}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: '#f44336',
                        cursor: 'pointer',
                        fontSize: '16px',
                        padding: '4px'
                      }}
                      title="Remove file"
                    >
                      ×
                    </button>
                  </div>
                  
                  {/* Variable type breakdown */}
                  {fileSchema && (
                    <div style={{ 
                      fontSize: '10px', 
                      color: '#2e7d32',
                      backgroundColor: '#f1f8e9',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid #c8e6c9'
                    }}>
                      <strong>Variable Analysis:</strong><br />
                      🔵 {fileSchema.filter(col => guessVariableType(col) === 'treatment').length} Treatment(s) • 
                      🔴 {fileSchema.filter(col => guessVariableType(col) === 'outcome').length} Outcome(s) • 
                      🟠 {fileSchema.filter(col => guessVariableType(col) === 'confounder').length} Confounder(s) • 
                      ⚪ {fileSchema.filter(col => guessVariableType(col) === 'unknown').length} Unknown
                    </div>
                  )}
                </div>
              )}
              
            </div>
          )}
        </div>

        {/* Model training */}
        <div style={{ marginBottom: '25px' }}>
          <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#555' }}>Model Training</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <button
              onClick={trainModel}
              disabled={isTraining || nodes.length === 0 || (useCustomData && !uploadedFile)}
              style={{
                padding: '10px 12px',
                backgroundColor: isTraining || (useCustomData && !uploadedFile) ? '#9e9e9e' : '#2196f3',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: isTraining || nodes.length === 0 || (useCustomData && !uploadedFile) ? 'not-allowed' : 'pointer',
                fontSize: '12px',
                fontWeight: 'bold'
              }}
            >
              {isTraining ? 'Training...' : 
               useCustomData && uploadedFile ? `🚀 Train with ${uploadedFile.name}` :
               useCustomData ? '🚀 Train Model (Upload file first)' : 
               '🚀 Train with Default Data'}
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

        {/* Column Interpretation and Causal Analysis */}
        {fileSchema && columnInterpretation && (
          <div style={{ marginTop: '12px' }}>
            {!isInterpretationConfirmed ? (
              // Step 1: Column Interpretation
              <div>
                <div style={{ 
                  fontSize: '12px', 
                  fontWeight: 'bold',
                  color: '#1976d2',
                  marginBottom: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  🤖 AI分析：请确认列的含义
                  <span style={{ fontSize: '10px', color: '#666', fontWeight: 'normal' }}>
                    (可修改)
                  </span>
                </div>
                
                <div style={{ 
                  maxHeight: '200px',
                  overflowY: 'auto',
                  marginBottom: '12px'
                }}>
                  {Object.entries(columnInterpretation).map(([column, info]) => (
                    <div key={column} style={{
                      padding: '8px',
                      margin: '4px 0',
                      backgroundColor: info.confidence > 0.7 ? '#e8f5e8' : 
                                     info.confidence > 0.5 ? '#fff3e0' : '#ffebee',
                      border: '1px solid #e0e0e0',
                      borderRadius: '4px',
                      fontSize: '10px'
                    }}>
                      <div style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'space-between',
                        marginBottom: '4px'
                      }}>
                        <strong style={{ color: '#333' }}>{column}</strong>
                        <span style={{ 
                          fontSize: '8px', 
                          color: '#666',
                          backgroundColor: '#f5f5f5',
                          padding: '2px 6px',
                          borderRadius: '10px'
                        }}>
                          {Math.round(info.confidence * 100)}% 确信度
                        </span>
                      </div>
                      
                      <div style={{ color: '#666', marginBottom: '6px' }}>
                        {info.description}
                      </div>
                      
                      <select
                        value={info.type}
                        onChange={(e) => updateColumnInterpretation(column, e.target.value)}
                        style={{
                          fontSize: '9px',
                          padding: '2px 4px',
                          border: '1px solid #ddd',
                          borderRadius: '3px',
                          backgroundColor: 'white',
                          width: '100%'
                        }}
                      >
                        <option value="treatment">Treatment (营销/干预变量)</option>
                        <option value="outcome">Outcome (结果变量)</option>
                        <option value="confounder">Confounder (混淆因子)</option>
                        <option value="time">Time (时间变量)</option>
                        <option value="unknown">Unknown (未知类型)</option>
                      </select>
                    </div>
                  ))}
                </div>
                
                <button
                  onClick={confirmInterpretation}
                  style={{
                    width: '100%',
                    padding: '8px',
                    backgroundColor: '#4caf50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontWeight: 'bold',
                    cursor: 'pointer'
                  }}
                >
                  ✅ 确认分析，生成因果建议
                </button>
              </div>
            ) : (
              // Step 2: Causal Analysis Results
              <div>
                <div style={{ 
                  fontSize: '12px', 
                  fontWeight: 'bold',
                  color: '#2e7d32',
                  marginBottom: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  ✅ 因果分析完成
                  <button
                    onClick={() => {
                      setIsInterpretationConfirmed(false);
                      setCausalAnalysis(null);
                    }}
                    style={{
                      fontSize: '8px',
                      padding: '2px 6px',
                      backgroundColor: '#ff9800',
                      color: 'white',
                      border: 'none',
                      borderRadius: '10px',
                      cursor: 'pointer'
                    }}
                  >
                    重新分析
                  </button>
                </div>
                
                {causalAnalysis && (
                  <div style={{
                    maxHeight: '250px',
                    overflowY: 'auto',
                    padding: '10px',
                    backgroundColor: '#f8f9fa',
                    border: '1px solid #e9ecef',
                    borderRadius: '4px',
                    fontSize: '9px',
                    lineHeight: '1.4',
                    whiteSpace: 'pre-wrap'
                  }}>
                    {causalAnalysis}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Draggable Splitter */}
      <div
        style={{
          width: '6px',
          background: isDragging ? '#2196f3' : '#ddd',
          cursor: 'col-resize',
          flexShrink: 0,
          position: 'relative',
          transition: isDragging ? 'none' : 'background 0.2s',
        }}
        onMouseDown={handleMouseDown}
      >
        {/* Visual indicator */}
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '2px',
            height: '30px',
            background: isDragging ? '#ffffff' : '#999',
            borderRadius: '1px',
            opacity: 0.7,
          }}
        />
      </div>

      {/* Main canvas area */}
      <div ref={reactFlowWrapper} style={{ 
        flex: 1, // Takes remaining space
        position: 'relative' 
      }}>
        <ReactFlow
          nodes={nodes.map(node => {
            // Check if node is in user's uploaded data schema
            let isInUserData = true;
            if (fileSchema) {
              isInUserData = fileSchema.includes(node.id) || fileSchema.includes(node.data?.label);
            }
            
            // Check if this node is a confounder for border styling
            const nodeId = node.data?.label || node.id;
            const nodeType = columnInterpretation && columnInterpretation[nodeId] 
              ? columnInterpretation[nodeId].type 
              : guessVariableType(nodeId);
            const isConfounder = nodeType === 'confounder';
            
            // Apply confounder border styling
            const confounderStyle = isConfounder ? {
              borderStyle: 'dashed',
              borderWidth: '2px'
            } : {};
            
            return {
              ...node,
              style: {
                ...node.style,
                ...confounderStyle,
                boxShadow: selectedNodes.includes(node.id) ? '0 0 0 2px #2196f3' : 'none'
              }
            };
          })}
          edges={edges.map(edge => {
            // Check if source and target nodes are in user's uploaded data schema
            const sourceNode = nodes.find(n => n.id === edge.source);
            const targetNode = nodes.find(n => n.id === edge.target);
            
            let isSourceInUserData = true;
            let isTargetInUserData = true;
            
            if (fileSchema && sourceNode) {
              isSourceInUserData = fileSchema.includes(sourceNode.id) || fileSchema.includes(sourceNode.data?.label);
            }
            
            if (fileSchema && targetNode) {
              isTargetInUserData = fileSchema.includes(targetNode.id) || fileSchema.includes(targetNode.data?.label);
            }
            
            // Check if source or target is a confounder
            const getNodeType = (nodeId: string, nodeLabel?: string) => {
              const id = nodeLabel || nodeId;
              if (columnInterpretation && columnInterpretation[id]) {
                return columnInterpretation[id].type;
              }
              return guessVariableType(id);
            };
            
            const isSourceConfounder = getNodeType(edge.source, edge.source) === 'confounder';
            const isTargetConfounder = getNodeType(edge.target, edge.target) === 'confounder';
            
            // Use dashed line ONLY if EITHER source OR target is a confounder
            // Mediators and other theoretical variables should use solid lines for causal pathways
            const shouldUseDashedLine = isSourceConfounder || isTargetConfounder;
            
            // Make user data connections more prominent
            const strokeWidth = isSourceInUserData && isTargetInUserData ? 3 : 2;
            
            return {
              ...edge,
              style: {
                ...edge.style,
                stroke: selectedEdges.includes(edge.id) ? '#2196f3' : edge.style?.stroke,
                strokeWidth: selectedEdges.includes(edge.id) ? 4 : strokeWidth,
                strokeDasharray: shouldUseDashedLine ? '8,4' : 'none'
              }
            };
          })}
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
