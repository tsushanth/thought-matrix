from fastapi import FastAPI, HTTPException
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import uuid
from datetime import datetime
from typing import Optional
import logging
from config import Config
from pydantic import BaseModel
from huggingface_hub import login


# Initialize configuration
cfg = Config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Thought Evaluator Service",
    description="LLM inference with thought pattern tracking",
    version="1.0.0"
)
login(token=cfg.HF_TOKEN)

class ThoughtEvaluator:
    _model_cache = {}

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.activation_hooks = []
        
    def _load_model(self):  # Changed from classmethod to instance method
        """Cache models to avoid repeated loading"""
        if cfg.MODEL_NAME not in self._model_cache:
            logger.info(f"Loading model: {cfg.MODEL_NAME}")
            self._model_cache[cfg.MODEL_NAME] = AutoModelForCausalLM.from_pretrained(
                cfg.MODEL_NAME,
                token=cfg.HF_TOKEN,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
        return self._model_cache[cfg.MODEL_NAME]

    def _setup_partitions(self):
        try:
            # Collect all relevant layers
            layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    layers.append(module)
                elif any(x in name for x in ['attn', 'mlp', 'ffn']):  # Common GPT layer patterns
                    layers.append(module)
            
            # Ensure we have enough layers
            required = cfg.N_PARTITIONS ** 2
            if len(layers) < required:
                layers = (layers * ((required // len(layers)) + 1))[:required]
            
            self.partitions = np.array_split(layers, required)
            logger.info(f"Created {len(self.partitions)} partitions")
            
        except Exception as e:
            logger.error(f"Partitioning failed: {str(e)}")
            raise

    def _attach_hooks(self):
        """Attach hooks to all partitions"""
        self.mind_map = np.zeros((cfg.N_PARTITIONS, cfg.N_PARTITIONS))
        self.activation_hooks = []
        
        for i in range(cfg.N_PARTITIONS):
            for j in range(cfg.N_PARTITIONS):
                # Handle both single modules and module lists
                target = self.partitions[i][j]
                modules = target if isinstance(target, (list, torch.nn.ModuleList)) else [target]
                
                for module in modules:
                    # Create closure with current i,j values
                    def make_hook(i, j):
                        def hook(module, input, output):
                            # Handle cases where output is a tuple (common in transformers)
                            actual_output = output[0] if isinstance(output, tuple) else output
                            self.mind_map[i,j] += actual_output.abs().mean().item()
                        return hook
                    
                    # Need to bind current i,j values to avoid late binding issue
                    hook = make_hook(i, j)
                    self.activation_hooks.append(
                        module.register_forward_hook(hook)
                    )
    
    async def infer(self, prompt: str, max_tokens: Optional[int] = None):
        """Run inference with thought monitoring"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            self._setup_partitions()
            self._attach_hooks()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or cfg.MAX_TOKENS,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=cfg.TEMPERATURE
                )
            
            # Normalize mind map
            self.mind_map = self.mind_map / np.max(self.mind_map)
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True), self.mind_map
        finally:
            # Ensure hooks are always removed
            for hook in self.activation_hooks:
                hook.remove()
            self.activation_hooks.clear()

def save_to_gcs(prompt: str, output: str, mind_map: np.ndarray) -> str:
    """Save results to Google Cloud Storage"""
    try:
        storage_client = storage.Client(credentials=cfg.gcp_auth['credentials'])
        bucket = storage_client.bucket(cfg.BUCKET_NAME)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"mindmaps/{timestamp}_{uuid.uuid4()}.npz"
        
        # Create in-memory buffer
        with io.BytesIO() as buffer:
            np.savez_compressed(
                buffer,
                prompt=prompt,
                output=output,
                mind_map=mind_map,
                model=cfg.MODEL_NAME,
                config=cfg.N_PARTITIONS
            )
            buffer.seek(0)  # Rewind the buffer
            blob = bucket.blob(filename)
            blob.upload_from_file(buffer, content_type='application/octet-stream')
        
        logger.info(f"Saved mind map to gs://{cfg.BUCKET_NAME}/{filename}")
        return filename
    except Exception as e:  # Broadened exception handling
        logger.error(f"GCS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save results: {str(e)}")
    
class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None

@app.post("/infer")
async def infer_endpoint(request: InferenceRequest):
    try:
        evaluator = ThoughtEvaluator()
        output, mind_map = await evaluator.infer(request.prompt, request.max_tokens)
        print(f"Output: {output}")
        print(f"Mind Map: {mind_map}")
        print(f"Save to GCS: {cfg.BUCKET_NAME}")
        file_id = save_to_gcs(request.prompt, output, mind_map)
        
        return {
            "output": output,
            "mind_map_id": file_id,
            "model": cfg.MODEL_NAME,
            "partitions": cfg.N_PARTITIONS,
            "normalized": True
        }
    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": cfg.MODEL_NAME,
        "gcp_bucket": cfg.BUCKET_NAME
    }