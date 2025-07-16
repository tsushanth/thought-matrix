from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Make Google Cloud optional
try:
    from google.cloud import storage
    from google.api_core.exceptions import GoogleAPIError
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    print("Warning: Google Cloud not available. Results will be saved locally.")
    GOOGLE_CLOUD_AVAILABLE = False
    storage = None
    GoogleAPIError = Exception

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import io
import numpy as np
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from config import Config
from pydantic import BaseModel, Field
from huggingface_hub import login
import json
import asyncio
from contextlib import asynccontextmanager

# Import our enhanced components
from enhanced_thought_matrix import (
    EnhancedThoughtMatrix, 
    ResearchExperiment, 
    QueryAnalyzer,
    ResearchMetrics
)

# Initialize configuration
cfg = Config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model caching
MODEL_CACHE = {}
TOKENIZER_CACHE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Enhanced Thought Matrix Service")
    login(token=cfg.HF_TOKEN)
    
    # Pre-load model and tokenizer
    try:
        await load_model_and_tokenizer()
        logger.info("Model and tokenizer pre-loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Thought Matrix Service")

app = FastAPI(
    title="Enhanced Thought Matrix Service",
    description="Advanced LLM inference with comprehensive thought pattern analysis",
    version="2.0.0",
    lifespan=lifespan
)

async def load_model_and_tokenizer():
    """Load and cache model and tokenizer"""
    global MODEL_CACHE, TOKENIZER_CACHE
    
    if cfg.MODEL_NAME not in MODEL_CACHE:
        logger.info(f"Loading model: {cfg.MODEL_NAME}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_NAME,
            token=cfg.HF_TOKEN,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, token=cfg.HF_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
        
        MODEL_CACHE[cfg.MODEL_NAME] = model
        TOKENIZER_CACHE[cfg.MODEL_NAME] = tokenizer
        
        logger.info(f"Model loaded on device: {device}")
    
    return MODEL_CACHE[cfg.MODEL_NAME], TOKENIZER_CACHE[cfg.MODEL_NAME]

class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for the model")
    max_tokens: Optional[int] = Field(100, description="Maximum tokens to generate")
    n_partitions: Optional[int] = Field(4, description="Number of partitions for thought matrix")
    include_detailed_analysis: Optional[bool] = Field(True, description="Include comprehensive research metrics")

class ExperimentRequest(BaseModel):
    max_tokens: Optional[int] = Field(50, description="Maximum tokens per query")
    n_partitions: Optional[int] = Field(4, description="Number of partitions for thought matrix")
    categories: Optional[List[str]] = Field(None, description="Specific categories to test (if None, all categories)")
    custom_queries: Optional[Dict[str, List[str]]] = Field(None, description="Custom queries by category")

class BatchInferenceRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of prompts to process")
    max_tokens: Optional[int] = Field(100, description="Maximum tokens per prompt")
    n_partitions: Optional[int] = Field(4, description="Number of partitions for thought matrix")

async def save_to_gcs(data: Dict[str, Any], prefix: str = "results") -> str:
    """Save results to Google Cloud Storage with enhanced error handling"""
    
    # If GCS not available, save locally
    if not GOOGLE_CLOUD_AVAILABLE or not cfg.gcp_auth.get('available', False):
        return save_locally(data, prefix)
    
    try:
        storage_client = storage.Client(credentials=cfg.gcp_auth.get('credentials'))
        bucket = storage_client.bucket(cfg.BUCKET_NAME)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{prefix}/{timestamp}_{uuid.uuid4()}.json"
        
        # Serialize data with numpy array handling
        def serialize_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, ResearchMetrics):
                return {
                    'activation_entropy': obj.activation_entropy,
                    'region_specialization': obj.region_specialization,
                    'cross_region_connectivity': obj.cross_region_connectivity,
                    'pattern_complexity': obj.pattern_complexity,
                    'category_signature': obj.category_signature.tolist(),
                    'temporal_dynamics': obj.temporal_dynamics,
                    'functional_distribution': obj.functional_distribution
                }
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # Create JSON string
        json_data = json.dumps(data, default=serialize_numpy, indent=2)
        
        # Upload to GCS
        blob = bucket.blob(filename)
        blob.upload_from_string(json_data, content_type='application/json')
        
        logger.info(f"Saved results to gs://{cfg.BUCKET_NAME}/{filename}")
        return filename
        
    except Exception as e:
        logger.warning(f"GCS save failed: {str(e)}. Falling back to local storage.")
        return save_locally(data, prefix)

def save_locally(data: Dict[str, Any], prefix: str = "results") -> str:
    """Save results locally when GCS is not available"""
    import os
    
    # Create local results directory
    local_dir = Path("./local_results")
    local_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}_{timestamp}_{uuid.uuid4()}.json"
    local_path = local_dir / filename
    
    # Serialize data with numpy array handling
    def serialize_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, ResearchMetrics):
            return {
                'activation_entropy': obj.activation_entropy,
                'region_specialization': obj.region_specialization,
                'cross_region_connectivity': obj.cross_region_connectivity,
                'pattern_complexity': obj.pattern_complexity,
                'category_signature': obj.category_signature.tolist(),
                'temporal_dynamics': obj.temporal_dynamics,
                'functional_distribution': obj.functional_distribution
            }
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(local_path, 'w') as f:
        json.dump(data, f, indent=2, default=serialize_numpy)
    
    logger.info(f"Saved results locally to: {local_path}")
    return str(local_path)

@app.post("/infer")
async def enhanced_inference(request: InferenceRequest):
    """Enhanced inference with comprehensive thought pattern analysis"""
    try:
        model, tokenizer = await load_model_and_tokenizer()
        
        # Create enhanced thought matrix
        thought_matrix = EnhancedThoughtMatrix(model, request.n_partitions)
        
        # Run inference with analysis
        result = await thought_matrix.infer_with_analysis(
            request.prompt, 
            tokenizer, 
            request.max_tokens
        )
        
        # Save to GCS
        file_id = await save_to_gcs(result, "single_inference")
        gcs_uri = f"gs://{cfg.BUCKET_NAME}/{file_id}"
        
        # Prepare response
        response = {
            "prompt": result['prompt'],
            "output": result['output'],
            "query_category": result['query_category'],
            "mind_map_shape": result['mind_map'].shape,
            "file_id": file_id,
            "gcs_uri": gcs_uri,
            "model": cfg.MODEL_NAME,
            "partitions": request.n_partitions,
            "timestamp": result['timestamp']
        }
        
        if request.include_detailed_analysis:
            response.update({
                "mind_map": result['mind_map'].tolist(),
                "metrics": {
                    'activation_entropy': result['metrics'].activation_entropy,
                    'region_specialization': result['metrics'].region_specialization,
                    'cross_region_connectivity': result['metrics'].cross_region_connectivity,
                    'pattern_complexity': result['metrics'].pattern_complexity,
                    'temporal_dynamics': result['metrics'].temporal_dynamics,
                    'functional_distribution': result['metrics'].functional_distribution
                },
                "functional_regions": result['functional_regions']
            })
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiment")
async def run_research_experiment(request: ExperimentRequest):
    """Run comprehensive research experiment across query categories"""
    try:
        model, tokenizer = await load_model_and_tokenizer()
        
        # Create research experiment
        experiment = ResearchExperiment(model, tokenizer, request.n_partitions)
        
        # Override test queries if custom ones provided
        if request.custom_queries:
            experiment.query_analyzer.categories.update(request.custom_queries)
        
        # Run experiment
        logger.info("Starting research experiment...")
        results = await experiment.run_experiment(request.max_tokens)
        logger.info(f"Experiment completed with {len(results['individual_results'])} queries")
        
        # Save comprehensive results
        file_id = await save_to_gcs(results, "research_experiment")
        gcs_uri = f"gs://{cfg.BUCKET_NAME}/{file_id}"
        
        # Generate summary statistics
        summary = {
            "experiment_overview": {
                "total_queries": results['experiment_metadata']['total_queries'],
                "categories_tested": results['experiment_metadata']['categories'],
                "partitions_used": request.n_partitions,
                "model": cfg.MODEL_NAME
            },
            "category_statistics": {},
            "cross_category_insights": results['cross_category_metrics'],
            "file_id": file_id,
            "gcs_uri": gcs_uri,
            "timestamp": results['experiment_metadata']['timestamp']
        }
        
        # Calculate category-level statistics
        for category, analysis in results['pattern_analysis'].items():
            summary["category_statistics"][category] = {
                "sample_size": analysis['sample_size'],
                "pattern_consistency": analysis['pattern_consistency'],
                "unique_regions_count": len(analysis['unique_regions']),
                "mean_activation_strength": float(np.mean(analysis['mean_pattern']))
            }
        
        return summary
        
    except Exception as e:
        logger.error(f"Research Experiment Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_infer")
async def batch_inference(request: BatchInferenceRequest):
    """Process multiple prompts in batch with thought pattern analysis"""
    try:
        model, tokenizer = await load_model_and_tokenizer()
        
        # Create thought matrix
        thought_matrix = EnhancedThoughtMatrix(model, request.n_partitions)
        
        results = []
        for i, prompt in enumerate(request.prompts):
            logger.info(f"Processing batch item {i+1}/{len(request.prompts)}")
            
            result = await thought_matrix.infer_with_analysis(
                prompt, 
                tokenizer, 
                request.max_tokens
            )
            
            # Simplify result for batch processing
            simplified_result = {
                "prompt": result['prompt'],
                "output": result['output'],
                "query_category": result['query_category'],
                "mind_map": result['mind_map'].tolist(),
                "activation_entropy": result['metrics'].activation_entropy,
                "pattern_complexity": result['metrics'].pattern_complexity,
                "timestamp": result['timestamp']
            }
            
            results.append(simplified_result)
        
        # Analyze batch patterns
        batch_analysis = {
            "total_processed": len(results),
            "category_distribution": {},
            "average_metrics": {
                "activation_entropy": np.mean([r['activation_entropy'] for r in results]),
                "pattern_complexity": np.mean([r['pattern_complexity'] for r in results])
            }
        }
        
        # Count categories
        for result in results:
            category = result['query_category']
            batch_analysis["category_distribution"][category] = \
                batch_analysis["category_distribution"].get(category, 0) + 1
        
        # Save batch results
        batch_data = {
            "batch_results": results,
            "batch_analysis": batch_analysis,
            "metadata": {
                "model": cfg.MODEL_NAME,
                "partitions": request.n_partitions,
                "max_tokens": request.max_tokens,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        file_id = await save_to_gcs(batch_data, "batch_inference")
        gcs_uri = f"gs://{cfg.BUCKET_NAME}/{file_id}"
        
        return {
            "batch_analysis": batch_analysis,
            "file_id": file_id,
            "gcs_uri": gcs_uri,
            "individual_results": results
        }
        
    except Exception as e:
        logger.error(f"Batch Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze_category/{category}")
async def analyze_query_category(category: str, sample_size: int = 10):
    """Analyze a specific query category with sample queries"""
    try:
        model, tokenizer = await load_model_and_tokenizer()
        
        # Create experiment with focus on specific category
        experiment = ResearchExperiment(model, tokenizer)
        test_queries = experiment.generate_test_queries()
        
        if category not in test_queries:
            raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
        
        # Limit to sample size
        queries = test_queries[category][:sample_size]
        
        results = []
        for query in queries:
            result = await experiment.thought_matrix.infer_with_analysis(
                query, tokenizer, max_tokens=50
            )
            results.append(result)
        
        # Analyze patterns for this category
        category_analysis = experiment.query_analyzer.analyze_category_patterns(results)
        
        # Save category-specific analysis
        analysis_data = {
            "category": category,
            "sample_size": len(results),
            "queries_analyzed": queries,
            "pattern_analysis": category_analysis.get(category, {}),
            "individual_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        file_id = await save_to_gcs(analysis_data, f"category_analysis_{category}")
        gcs_uri = f"gs://{cfg.BUCKET_NAME}/{file_id}"
        
        return {
            "category": category,
            "analysis": category_analysis.get(category, {}),
            "sample_queries": queries,
            "file_id": file_id,
            "gcs_uri": gcs_uri
        }
        
    except Exception as e:
        logger.error(f"Category Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def get_model_info():
    """Get information about the currently loaded model"""
    try:
        model, tokenizer = await load_model_and_tokenizer()
        
        # Get model architecture info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Analyze model structure for functional regions
        thought_matrix = EnhancedThoughtMatrix(model, n_partitions=4)
        functional_info = {}
        
        for func_type, layers in thought_matrix.functional_regions.items():
            functional_info[func_type] = {
                "layer_count": len(layers),
                "layer_names": [name for name, _ in layers[:5]]  # First 5 for brevity
            }
        
        return {
            "model_name": cfg.MODEL_NAME,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(next(model.parameters()).device),
            "model_type": model.config.model_type if hasattr(model, 'config') else "unknown",
            "functional_regions": functional_info,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "max_position_embeddings": getattr(model.config, 'max_position_embeddings', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"Model Info Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check with model status"""
    try:
        model_loaded = cfg.MODEL_NAME in MODEL_CACHE
        tokenizer_loaded = cfg.MODEL_NAME in TOKENIZER_CACHE
        
        if model_loaded and tokenizer_loaded:
            model = MODEL_CACHE[cfg.MODEL_NAME]
            device = str(next(model.parameters()).device)
        else:
            device = "not_loaded"
        
        return {
            "status": "healthy",
            "model_name": cfg.MODEL_NAME,
            "model_loaded": model_loaded,
            "tokenizer_loaded": tokenizer_loaded,
            "device": device,
            "gcp_bucket": cfg.BUCKET_NAME,
            "cuda_available": torch.cuda.is_available(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Enhanced Thought Matrix Service",
        "version": "2.0.0",
        "description": "Advanced LLM inference with comprehensive thought pattern analysis",
        "endpoints": {
            "POST /infer": "Single inference with thought analysis",
            "POST /experiment": "Run comprehensive research experiment",
            "POST /batch_infer": "Batch inference processing",
            "GET /analyze_category/{category}": "Analyze specific query category",
            "GET /model_info": "Get model architecture information",
            "GET /health": "Health check endpoint"
        },
        "model": cfg.MODEL_NAME,
        "documentation": "/docs"
    }