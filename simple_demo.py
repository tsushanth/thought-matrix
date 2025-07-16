#!/usr/bin/env python3
"""
Simple Local Demo - No Cloud Dependencies
=========================================

This script runs a basic thought matrix experiment using GPT-2
without requiring HuggingFace tokens or Google Cloud.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleThoughtMatrix:
    """Simplified thought matrix for local demo"""
    
    def __init__(self, model, n_partitions=3):
        self.model = model
        self.n_partitions = n_partitions
        self.device = next(model.parameters()).device
        self.mind_map = np.zeros((n_partitions, n_partitions))
        self.hooks = []
        
    def _get_layers(self):
        """Get model layers for partitioning"""
        layers = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and len(list(module.parameters())) > 0:
                layers.append(module)
        return layers
    
    def _setup_hooks(self):
        """Setup activation tracking hooks"""
        layers = self._get_layers()
        total_partitions = self.n_partitions ** 2
        
        if len(layers) < total_partitions:
            # Repeat layers if we don't have enough
            layers = (layers * ((total_partitions // len(layers)) + 1))[:total_partitions]
        
        # Distribute layers across partitions
        layers_per_partition = len(layers) // total_partitions
        
        self.mind_map.fill(0)
        
        idx = 0
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                start_idx = idx * layers_per_partition
                end_idx = min((idx + 1) * layers_per_partition, len(layers))
                
                for layer_idx in range(start_idx, end_idx):
                    if layer_idx < len(layers):
                        def make_hook(i, j):
                            def hook(module, input, output):
                                try:
                                    if isinstance(output, tuple):
                                        output = output[0]
                                    activation = torch.abs(output).mean().item()
                                    self.mind_map[i, j] += activation
                                except:
                                    pass
                            return hook
                        
                        hook_fn = make_hook(i, j)
                        handle = layers[layer_idx].register_forward_hook(hook_fn)
                        self.hooks.append(handle)
                
                idx += 1
    
    def _cleanup_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    async def infer_with_analysis(self, prompt, tokenizer, max_tokens=50):
        """Run inference with thought tracking"""
        try:
            # Setup hooks
            self._setup_hooks()
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Normalize mind map
            if self.mind_map.sum() > 0:
                self.mind_map = self.mind_map / self.mind_map.max()
            
            # Decode output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Simple category detection
            category = self._simple_categorize(prompt)
            
            # Calculate simple metrics
            entropy = self._calculate_entropy(self.mind_map)
            complexity = np.std(self.mind_map)
            
            return {
                'prompt': prompt,
                'output': output_text,
                'category': category,
                'mind_map': self.mind_map.copy(),
                'entropy': entropy,
                'complexity': complexity,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            self._cleanup_hooks()
    
    def _simple_categorize(self, prompt):
        """Simple query categorization"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['what is', 'define', 'who is']):
            return 'factual'
        elif any(word in prompt_lower for word in ['why', 'how', 'explain']):
            return 'reasoning'
        elif any(word in prompt_lower for word in ['write', 'create', 'imagine']):
            return 'creative'
        elif any(word in prompt_lower for word in ['calculate', 'solve', '+', '-', '*']):
            return 'mathematical'
        else:
            return 'other'
    
    def _calculate_entropy(self, matrix):
        """Calculate activation entropy"""
        flat = matrix.flatten()
        flat = flat / (flat.sum() + 1e-10)
        return -np.sum(flat * np.log2(flat + 1e-10))

class SimpleExperiment:
    """Simple experiment runner"""
    
    def __init__(self):
        self.results = []
        
    def get_test_queries(self):
        """Get a small set of test queries"""
        return {
            'factual': [
                "What is the capital of France?",
                "Define artificial intelligence.",
                "Who invented the telephone?"
            ],
            'reasoning': [
                "Why do objects fall to the ground?",
                "How does photosynthesis work?",
                "Explain why the sky is blue."
            ],
            'creative': [
                "Write a short story about a robot.",
                "Create a poem about spring.",
                "Imagine life on Mars."
            ],
            'mathematical': [
                "Calculate 15 + 27.",
                "What is 25% of 100?",
                "Solve for x: 2x + 4 = 10."
            ]
        }
    
    async def run_experiment(self, model, tokenizer, max_tokens=50):
        """Run simple experiment"""
        thought_matrix = SimpleThoughtMatrix(model, n_partitions=3)
        test_queries = self.get_test_queries()
        
        results = []
        
        for category, queries in test_queries.items():
            logger.info(f"Testing category: {category}")
            
            for i, query in enumerate(queries):
                logger.info(f"  Query {i+1}/{len(queries)}: {query}")
                
                result = await thought_matrix.infer_with_analysis(
                    query, tokenizer, max_tokens
                )
                results.append(result)
        
        return results
    
    def analyze_results(self, results):
        """Simple analysis of results"""
        analysis = {
            'total_queries': len(results),
            'categories': {},
            'overall_stats': {}
        }
        
        # Group by category
        by_category = {}
        for result in results:
            cat = result['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)
        
        # Calculate category stats
        for category, cat_results in by_category.items():
            entropies = [r['entropy'] for r in cat_results]
            complexities = [r['complexity'] for r in cat_results]
            
            analysis['categories'][category] = {
                'count': len(cat_results),
                'avg_entropy': float(np.mean(entropies)),
                'avg_complexity': float(np.mean(complexities)),
                'sample_queries': [r['prompt'] for r in cat_results[:2]]
            }
        
        # Overall stats
        all_entropies = [r['entropy'] for r in results]
        all_complexities = [r['complexity'] for r in results]
        
        analysis['overall_stats'] = {
            'avg_entropy': float(np.mean(all_entropies)),
            'avg_complexity': float(np.mean(all_complexities)),
            'entropy_range': [float(min(all_entropies)), float(max(all_entropies))],
            'complexity_range': [float(min(all_complexities)), float(max(all_complexities))]
        }
        
        return analysis

async def main():
    """Run the simple demo"""
    print("="*60)
    print("SIMPLE THOUGHT MATRIX DEMO")
    print("="*60)
    print("Loading GPT-2 model (this may take a moment)...")
    
    # Load GPT-2 (no tokens required)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Model loaded on: {device}")
    
    # Run experiment
    experiment = SimpleExperiment()
    
    print("\nRunning thought matrix experiment...")
    results = await experiment.run_experiment(model, tokenizer, max_tokens=30)
    
    print(f"\nCompleted {len(results)} queries!")
    
    # Analyze results
    analysis = experiment.analyze_results(results)
    
    # Save results
    output_dir = Path("./demo_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"demo_results_{timestamp}.json"
    analysis_file = output_dir / f"demo_analysis_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    
    print(f"Total queries processed: {analysis['total_queries']}")
    print(f"Overall average entropy: {analysis['overall_stats']['avg_entropy']:.3f}")
    print(f"Overall average complexity: {analysis['overall_stats']['avg_complexity']:.3f}")
    
    print("\nBy Category:")
    for category, stats in analysis['categories'].items():
        print(f"  {category.upper()}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg Entropy: {stats['avg_entropy']:.3f}")
        print(f"    Avg Complexity: {stats['avg_complexity']:.3f}")
        print(f"    Sample: '{stats['sample_queries'][0]}'")
    
    print(f"\nDetailed results saved to: {output_dir}")
    
    # Show a sample mind map
    sample_result = results[0]
    print(f"\nSample Mind Map for: '{sample_result['prompt']}'")
    print("Matrix shape:", sample_result['mind_map'].shape)
    print("Activation pattern:")
    mind_map = sample_result['mind_map']
    for i in range(mind_map.shape[0]):
        row_str = " ".join([f"{val:.2f}" for val in mind_map[i]])
        print(f"  [{row_str}]")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())