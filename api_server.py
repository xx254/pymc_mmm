#!/usr/bin/env python3
"""
FastAPI backend service for Causal DAG Editor
Connects React frontend with Python Causal MMM model training
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

# Add current directory to Python path for importing causal_mmm_tutorial
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import CausalMMMTutorial
try:
    from causal_mmm_tutorial import CausalMMMTutorial
    CAUSAL_MMM_AVAILABLE = True
    print("✅ CausalMMMTutorial imported successfully")
except ImportError as e:
    print(f"⚠️  Warning: Could not import CausalMMMTutorial: {e}")
    CAUSAL_MMM_AVAILABLE = False
    
    # Create a fallback base class
    class CausalMMMTutorial:
        def __init__(self):
            self.df = None
            self.data = None
        
        def generate_synthetic_data(self):
            print("Simulated data generation (fallback mode)")
            return None
        
        def run_causal_model(self, version="full"):
            print(f"Simulated causal model training (fallback mode, version: {version})")
            return None
        
        def run_correlational_model(self):
            print("Simulated correlational model training (fallback mode)")
            return None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Causal DAG Editor API", version="1.0.0")

# Add CORS middleware to allow React frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data model definitions
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
    """Enhanced version of CausalMMMTutorial with dynamic DAG support"""
    
    def __init__(self):
        super().__init__()
        self.custom_dag = None
        self.custom_dag_dot = None
        
    def set_custom_dag(self, dag_structure: DAGStructure, dag_dot_string: str):
        """Set custom DAG structure"""
        self.custom_dag = dag_structure
        self.custom_dag_dot = dag_dot_string
        
    def create_dynamic_dag_string(self, dag_structure: DAGStructure) -> str:
        """Create DOT string based on DAG structure"""
        if not dag_structure.edges:
            return "digraph { }"
            
        dot_string = "digraph {\n"
        
        # Add node definitions (optional, for better visualization)
        for node in dag_structure.nodes:
            node_id = node.id.replace(' ', '_').replace('(', '').replace(')', '')
            dot_string += f'  {node_id} [label="{node.label}"];\n'
        
        # Add edges
        for edge in dag_structure.edges:
            source = edge.source.replace(' ', '_').replace('(', '').replace(')', '')
            target = edge.target.replace(' ', '_').replace('(', '').replace(')', '')
            dot_string += f"  {source} -> {target};\n"
        
        dot_string += "}"
        return dot_string
        
    def map_dag_to_model_variables(self, dag_structure: DAGStructure) -> Dict[str, Any]:
        """Map DAG structure to model variables"""
        # Identify different types of nodes
        treatment_nodes = []
        outcome_nodes = []
        control_nodes = []
        
        for node in dag_structure.nodes:
            node_label = node.label.lower()
            node_id = node.id.lower()
            
            # Identify node types based on node labels and IDs
            if any(keyword in node_label for keyword in ['x1', 'x2', 'social', 'search', 'marketing', 'ads', 'treatment']):
                treatment_nodes.append(node.id)
            elif any(keyword in node_label for keyword in ['target', 'sales', 'y', 'outcome']):
                outcome_nodes.append(node.id)
            elif any(keyword in node_label for keyword in ['christmas', 'holiday', 'competitor', 'market', 'confounder', 'unobserved', 'mediator']):
                control_nodes.append(node.id)
                
        return {
            'treatment_nodes': treatment_nodes,
            'outcome_nodes': outcome_nodes,
            'control_nodes': control_nodes,
            'channel_columns': [node for node in treatment_nodes if any(x in node.lower() for x in ['x1', 'x2'])],
            'outcome_node': outcome_nodes[0] if outcome_nodes else 'y'
        }
        
    def run_custom_model(self, dag_structure: DAGStructure, dag_type: str):
        """Run model based on custom DAG"""
        try:
            print(f"🔥 DEBUG: Starting custom model training, DAG type: {dag_type}")
            print(f"🔥 DEBUG: CAUSAL_MMM_AVAILABLE = {CAUSAL_MMM_AVAILABLE}")
            logger.info(f"Starting custom model training, DAG type: {dag_type}")
            
            # Set custom DAG
            self.set_custom_dag(dag_structure, self.create_dynamic_dag_string(dag_structure))
            
            # Map DAG to model variables
            model_mapping = self.map_dag_to_model_variables(dag_structure)
            print(f"🔥 DEBUG: Model mapping: {model_mapping}")
            logger.info(f"Model mapping: {model_mapping}")
            
            if not CAUSAL_MMM_AVAILABLE:
                # If real PyMC-Marketing is unavailable, return simulation results
                print("🔥 DEBUG: Using simulation mode for model training...")
                logger.info("Using simulation mode for model training...")
                
                model_summary = {
                    'dag_type': dag_type,
                    'nodes_count': len(dag_structure.nodes),
                    'edges_count': len(dag_structure.edges),
                    'treatment_variables': model_mapping['treatment_nodes'],
                    'outcome_variables': model_mapping['outcome_nodes'],
                    'control_variables': model_mapping['control_nodes'],
                    'mode': 'simulation'
                }
                
                print("🔥 DEBUG: Returning simulation success result")
                return {
                    'status': 'success',
                    'message': f'Model training completed (simulation mode)! Used DAG structure with {len(dag_structure.nodes)} nodes and {len(dag_structure.edges)} edges. Note: This is a simulation result, please install complete dependencies for real model training.',
                    'model_summary': model_summary,
                    'convergence_info': {
                        'r_hat_max': 1.01,  # Simulated convergence metrics
                        'ess_bulk_min': 1000,
                        'divergences': 0,
                        'mode': 'simulated'
                    }
                }
            
            print("🔥 DEBUG: Attempting real model training...")
            # Generate data (if not already generated)
            if self.df is None:
                print("🔥 DEBUG: Generating synthetic data...")
                logger.info("Generating synthetic data...")
                try:
                    self.generate_synthetic_data()
                    print("🔥 DEBUG: Data loading/generation successful")
                    print(f"🔥 DEBUG: Data shape: {self.df.shape}")
                    print(f"🔥 DEBUG: Target variable statistics: mean={self.df['target'].mean():.2f}, std={self.df['target'].std():.2f}")
                except Exception as e:
                    print(f"🔥 DEBUG: Data loading/generation failed: {e}")
                    print(f"🔥 DEBUG: Data loading/generation failure details: {traceback.format_exc()}")
                    raise
            
            # Choose training method based on DAG type
            print(f"🔥 DEBUG: Starting model training based on DAG type {dag_type}...")
            result = None
            if dag_type == 'business':
                # Use predefined business scenario model
                print("🔥 DEBUG: Running business scenario model...")
                try:
                    result = self.run_causal_model(version="full")
                    print(f"🔥 DEBUG: Business scenario model training result: {type(result)}")
                except Exception as e:
                    error_msg = f"Business scenario model training failed: {str(e)}"
                    error_traceback = traceback.format_exc()
                    print(f"🔥 DEBUG: {error_msg}")
                    print(f"🔥 DEBUG: Exception type: {type(e).__name__}")
                    print(f"🔥 DEBUG: Detailed error: {error_traceback}")
                    logger.error(error_msg)
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(error_traceback)
                    # Re-raise exception with more context
                    raise Exception(f"{error_msg} (Exception type: {type(e).__name__})") from e
            elif dag_type == 'simple':
                # Use simplified model
                print("🔥 DEBUG: Running simplified model...")
                try:
                    result = self.run_causal_model(version="simple")
                    print(f"🔥 DEBUG: Simplified model training result: {type(result)}")
                except Exception as e:
                    error_msg = f"Simplified model training failed: {str(e)}"
                    error_traceback = traceback.format_exc()
                    print(f"🔥 DEBUG: {error_msg}")
                    print(f"🔥 DEBUG: Exception type: {type(e).__name__}")
                    print(f"🔥 DEBUG: Detailed error: {error_traceback}")
                    logger.error(error_msg)
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(error_traceback)
                    # Re-raise exception with more context
                    raise Exception(f"{error_msg} (Exception type: {type(e).__name__})") from e
            else:
                # Custom model - use basic correlational model
                print("🔥 DEBUG: Running custom model (basic correlational model)...")
                logger.info("Running custom model (basic correlational model)...")
                try:
                    result = self.run_correlational_model()
                    print(f"🔥 DEBUG: Correlational model training result: {type(result)}")
                except Exception as e:
                    error_msg = f"Correlational model training failed: {str(e)}"
                    error_traceback = traceback.format_exc()
                    print(f"🔥 DEBUG: {error_msg}")
                    print(f"🔥 DEBUG: Exception type: {type(e).__name__}")
                    print(f"🔥 DEBUG: Detailed error: {error_traceback}")
                    logger.error(error_msg)
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(error_traceback)
                    # Re-raise exception with more context
                    raise Exception(f"{error_msg} (Exception type: {type(e).__name__})") from e
            
            if result is None:
                error_msg = "Model training result is None - possible causes: data issues, model configuration errors, or incomplete dependencies"
                print(f"🔥 DEBUG: {error_msg}")
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': f'PyMC-Marketing training failed: {error_msg}',
                    'error_details': {
                        'error_type': 'NullResult',
                        'dag_type': dag_type,
                        'nodes_count': len(dag_structure.nodes),
                        'edges_count': len(dag_structure.edges),
                        'causal_mmm_available': CAUSAL_MMM_AVAILABLE
                    }
                }
            
            print(f"🔥 DEBUG: Model training successful, result type: {type(result)}")
            
            # Generate model evaluation plots and metrics
            evaluation_result = None
            try:
                print("🔥 DEBUG: Starting model evaluation plot generation...")
                evaluation_result = self.generate_model_evaluation_plots(result)
                if evaluation_result:
                    print(f"🔥 DEBUG: Model evaluation completed - R²: {evaluation_result['r2_score']:.4f}, MAPE: {evaluation_result['mape']:.4f}")
                else:
                    print("🔥 DEBUG: Model evaluation result is empty")
            except Exception as e:
                print(f"🔥 DEBUG: Model evaluation failed: {e}")
                print(f"🔥 DEBUG: Model evaluation detailed error: {traceback.format_exc()}")
            
            # Prepare return result
            model_summary = {
                'dag_type': dag_type,
                'nodes_count': len(dag_structure.nodes),
                'edges_count': len(dag_structure.edges),
                'treatment_variables': model_mapping['treatment_nodes'],
                'outcome_variables': model_mapping['outcome_nodes'],
                'control_variables': model_mapping['control_nodes']
            }
            
            # If there are evaluation results, add them to model summary
            if evaluation_result:
                model_summary.update({
                    'fit_quality': {
                        'r2_score': evaluation_result['r2_score'],
                        'mape': evaluation_result['mape'],
                        'mae': evaluation_result['mae'],
                        'rmse': evaluation_result['rmse'],
                        'sample_size': evaluation_result['sample_size']
                    },
                    'data_info': {
                        'prediction_mean': evaluation_result['prediction_mean'],
                        'prediction_std': evaluation_result['prediction_std'],
                        'actual_mean': evaluation_result['actual_mean'],
                        'actual_std': evaluation_result['actual_std']
                    },
                    'plot_available': True,
                    'plot_path': evaluation_result['plot_path'],
                    'chart_data': evaluation_result.get('chart_data')  # Add chart data for frontend
                })
            else:
                model_summary['fit_quality'] = None
                model_summary['plot_available'] = False
            
            # Check model convergence
            convergence_info = {}
            if hasattr(result, 'idata') and result.idata is not None:
                try:
                    import arviz as az
                    convergence_info = {
                        'r_hat_max': float(az.rhat(result.idata).max()),
                        'ess_bulk_min': float(az.ess(result.idata).min()),
                        'divergences': int(result.idata["sample_stats"]["diverging"].sum())
                    }
                    print(f"🔥 DEBUG: Convergence metrics calculated successfully: {convergence_info}")
                except Exception as e:
                    print(f"🔥 DEBUG: Cannot calculate convergence metrics: {e}")
                    logger.warning(f"Cannot calculate convergence metrics: {e}")
            else:
                print("🔥 DEBUG: Model result has no idata attribute")
            
            print("🔥 DEBUG: Returning success result")
            return {
                'status': 'success',
                'message': f'Model training completed! Used DAG structure with {len(dag_structure.nodes)} nodes and {len(dag_structure.edges)} edges.',
                'model_summary': model_summary,
                'convergence_info': convergence_info
            }
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"🔥 DEBUG: Model training exception occurred: {str(e)}")
            print(f"🔥 DEBUG: Exception type: {type(e).__name__}")
            print(f"🔥 DEBUG: Detailed exception info: {error_traceback}")
            logger.error(f"Model training failed: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(error_traceback)
            
            # Build detailed error response
            error_response = {
                'status': 'error',
                'message': f'Model training failed: {str(e)}',
                'error_details': {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'traceback': error_traceback,
                    'dag_type': dag_type if 'dag_type' in locals() else 'unknown',
                    'causal_mmm_available': CAUSAL_MMM_AVAILABLE
                }
            }
            
            # If DAG structure info is available, include it
            if 'dag_structure' in locals() and dag_structure:
                error_response['error_details'].update({
                    'nodes_count': len(dag_structure.nodes),
                    'edges_count': len(dag_structure.edges)
                })
            
            return error_response

@app.get("/")
async def root():
    """Root path, returns API information"""
    return {
        "message": "Causal DAG Editor API",
        "version": "1.0.0",
        "causal_mmm_available": CAUSAL_MMM_AVAILABLE,
        "status": "running" if CAUSAL_MMM_AVAILABLE else "simulation mode",
        "endpoints": {
            "train_model": "/train-model",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "causal_mmm_available": CAUSAL_MMM_AVAILABLE,
        "mode": "full" if CAUSAL_MMM_AVAILABLE else "simulation"
    }

@app.post("/train-model", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Main endpoint for training causal models"""
    
    try:
        logger.info(f"Received training request, DAG type: {request.dag_type}")
        logger.info(f"Node count: {len(request.dag_structure.nodes)}")
        logger.info(f"Edge count: {len(request.dag_structure.edges)}")
        
        # Validate input
        if len(request.dag_structure.nodes) == 0:
            raise HTTPException(
                status_code=400,
                detail="DAG structure cannot be empty, please add at least one node"
            )
        
        # Create enhanced tutorial instance
        tutorial = EnhancedCausalMMMTutorial()
        
        # Run model training
        result = tutorial.run_custom_model(request.dag_structure, request.dag_type)
        
        return TrainingResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error occurred while training model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/validate-dag")
async def validate_dag(dag_structure: DAGStructure):
    """Validate DAG structure validity"""
    try:
        # Basic validation
        if len(dag_structure.nodes) == 0:
            return {"valid": False, "message": "DAG must contain at least one node"}
        
        # Check edge validity
        node_ids = {node.id for node in dag_structure.nodes}
        for edge in dag_structure.edges:
            if edge.source not in node_ids:
                return {"valid": False, "message": f"Edge source node '{edge.source}' does not exist"}
            if edge.target not in node_ids:
                return {"valid": False, "message": f"Edge target node '{edge.target}' does not exist"}
        
        # Check for cycles (simple check)
        # TODO: Implement more complex cycle detection algorithm
        
        return {"valid": True, "message": "DAG structure is valid"}
        
    except Exception as e:
        logger.error(f"Error occurred while validating DAG: {str(e)}")
        return {"valid": False, "message": f"Validation failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Causal DAG Editor API service...")
    print("📊 Frontend URL: http://localhost:3000")
    print("🔗 API URL: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print(f"🔧 Mode: {'Full functionality' if CAUSAL_MMM_AVAILABLE else 'Simulation mode'}")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 