import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

# Make Google Cloud optional for local development
try:
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    print("Warning: Google Cloud libraries not available. GCS features will be disabled.")
    GOOGLE_CLOUD_AVAILABLE = False
    service_account = None

@dataclass
class ModelConfig:
    """Model-specific configuration"""
    name: str
    max_tokens: int = 100
    temperature: float = 0.7
    n_partitions: int = 4
    
class Config:
    """Enhanced configuration for the thought matrix service"""
    
    def __init__(self):
        # Model Configuration
        self.MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/deepseek-coder-1.3b-instruct")
        self.HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "100"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
        self.N_PARTITIONS = int(os.getenv("N_PARTITIONS", "4"))
        
        # GCP Configuration
        self.BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "thought-matrix-results")
        self.GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
        self.GCP_SERVICE_ACCOUNT_PATH = os.getenv("GCP_SERVICE_ACCOUNT_PATH")
        
        # Research Configuration
        self.RESEARCH_CONFIG = {
            "default_sample_size": int(os.getenv("RESEARCH_SAMPLE_SIZE", "10")),
            "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", "50")),
            "enable_detailed_logging": os.getenv("ENABLE_DETAILED_LOGGING", "true").lower() == "true",
            "save_individual_results": os.getenv("SAVE_INDIVIDUAL_RESULTS", "true").lower() == "true",
            "auto_categorize_queries": os.getenv("AUTO_CATEGORIZE_QUERIES", "true").lower() == "true"
        }
        
        # Performance Configuration
        self.PERFORMANCE_CONFIG = {
            "use_mixed_precision": os.getenv("USE_MIXED_PRECISION", "true").lower() == "true",
            "gradient_checkpointing": os.getenv("GRADIENT_CHECKPOINTING", "false").lower() == "true",
            "max_memory_gb": int(os.getenv("MAX_MEMORY_GB", "16")),
            "batch_timeout_seconds": int(os.getenv("BATCH_TIMEOUT_SECONDS", "300"))
        }
        
        # Monitoring Configuration
        self.MONITORING_CONFIG = {
            "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
            "metrics_interval": int(os.getenv("METRICS_INTERVAL", "60")),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "enable_profiling": os.getenv("ENABLE_PROFILING", "false").lower() == "true"
        }
        
        # Initialize GCP authentication
        self.gcp_auth = self._setup_gcp_auth()
        
        # Validate configuration
        self._validate_config()
    
    def _setup_gcp_auth(self) -> Dict[str, Any]:
        """Setup GCP authentication"""
        auth_config = {'available': GOOGLE_CLOUD_AVAILABLE}
        
        if not GOOGLE_CLOUD_AVAILABLE:
            auth_config['method'] = 'unavailable'
            return auth_config
        
        if self.GCP_SERVICE_ACCOUNT_PATH and os.path.exists(self.GCP_SERVICE_ACCOUNT_PATH):
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    self.GCP_SERVICE_ACCOUNT_PATH
                )
                auth_config['credentials'] = credentials
                auth_config['method'] = 'service_account_file'
            except Exception as e:
                print(f"Warning: Failed to load service account file: {e}")
                auth_config['method'] = 'default'
        else:
            # Try to use default credentials (for GCP environments)
            auth_config['method'] = 'default'
        
        return auth_config
    
    def _validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        # Required environment variables
        if not self.HF_TOKEN:
            errors.append("HUGGINGFACE_TOKEN environment variable is required")
        
        if not self.BUCKET_NAME:
            errors.append("GCS_BUCKET_NAME environment variable is required")
        
        # Validate numeric ranges
        if self.N_PARTITIONS < 2 or self.N_PARTITIONS > 10:
            errors.append("N_PARTITIONS must be between 2 and 10")
        
        if self.TEMPERATURE < 0 or self.TEMPERATURE > 2:
            errors.append("TEMPERATURE must be between 0 and 2")
        
        if self.MAX_TOKENS < 1 or self.MAX_TOKENS > 2048:
            errors.append("MAX_TOKENS must be between 1 and 2048")
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    def get_model_config(self) -> ModelConfig:
        """Get model-specific configuration"""
        return ModelConfig(
            name=self.MODEL_NAME,
            max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE,
            n_partitions=self.N_PARTITIONS
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": {
                "name": self.MODEL_NAME,
                "max_tokens": self.MAX_TOKENS,
                "temperature": self.TEMPERATURE,
                "n_partitions": self.N_PARTITIONS
            },
            "gcp": {
                "bucket_name": self.BUCKET_NAME,
                "project_id": self.GCP_PROJECT_ID,
                "auth_method": self.gcp_auth.get('method', 'unknown')
            },
            "research": self.RESEARCH_CONFIG,
            "performance": self.PERFORMANCE_CONFIG,
            "monitoring": self.MONITORING_CONFIG
        }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Set environment variables from config file
        for section, settings in config_data.items():
            if isinstance(settings, dict):
                for key, value in settings.items():
                    env_key = f"{section.upper()}_{key.upper()}"
                    os.environ[env_key] = str(value)
            else:
                os.environ[section.upper()] = str(settings)
        
        return cls()
    
    def save_to_file(self, config_path: str):
        """Save current configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Model-specific configurations for different research scenarios
MODEL_CONFIGS = {
    "deepseek-coder-1.3b": ModelConfig(
        name="deepseek-ai/deepseek-coder-1.3b-instruct",
        max_tokens=150,
        temperature=0.7,
        n_partitions=4
    ),
    "deepseek-coder-6.7b": ModelConfig(
        name="deepseek-ai/deepseek-coder-6.7b-instruct",
        max_tokens=200,
        temperature=0.6,
        n_partitions=6
    ),
    "deepseek-llm-7b": ModelConfig(
        name="deepseek-ai/deepseek-llm-7b-chat",
        max_tokens=200,
        temperature=0.8,
        n_partitions=6
    )
}

# Research experiment templates
EXPERIMENT_TEMPLATES = {
    "basic_cognitive": {
        "description": "Basic cognitive assessment across all categories",
        "categories": ["factual", "reasoning", "creative", "mathematical", "ethical"],
        "sample_size_per_category": 10,
        "max_tokens": 100
    },
    "reasoning_focus": {
        "description": "Deep analysis of reasoning capabilities",
        "categories": ["reasoning", "mathematical", "ethical"],
        "sample_size_per_category": 20,
        "max_tokens": 150
    },
    "creativity_analysis": {
        "description": "Comprehensive creativity assessment",
        "categories": ["creative", "conversational"],
        "sample_size_per_category": 15,
        "max_tokens": 200
    },
    "comprehensive": {
        "description": "Full comprehensive analysis for research paper",
        "categories": ["factual", "reasoning", "creative", "mathematical", "ethical", "conversational"],
        "sample_size_per_category": 25,
        "max_tokens": 150
    }
}

def get_experiment_config(template_name: str) -> Dict[str, Any]:
    """Get configuration for a specific experiment template"""
    if template_name not in EXPERIMENT_TEMPLATES:
        raise ValueError(f"Unknown experiment template: {template_name}")
    
    return EXPERIMENT_TEMPLATES[template_name].copy()

def create_custom_experiment(categories: List[str], 
                           sample_size: int = 10, 
                           max_tokens: int = 100,
                           description: str = "Custom experiment") -> Dict[str, Any]:
    """Create a custom experiment configuration"""
    return {
        "description": description,
        "categories": categories,
        "sample_size_per_category": sample_size,
        "max_tokens": max_tokens
    }