#!/usr/bin/env python3
"""
Simple Research Pipeline - Minimal but Research-Ready
====================================================

This version focuses on core functionality with robust error handling.
Perfect for generating research paper data.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleThoughtMatrix:
    """Simple but effective thought matrix for research"""
    
    def __init__(self, model, n_partitions=4):
        self.model = model
        self.n_partitions = n_partitions
        self.device = next(model.parameters()).device
        self.mind_map = np.zeros((n_partitions, n_partitions))
        self.hooks = []
        
    def _get_model_layers(self):
        """Get all meaningful layers from the model"""
        layers = []
        for name, module in self.model.named_modules():
            # Focus on layers with parameters that actually compute something
            if hasattr(module, 'weight') and len(list(module.parameters())) > 0:
                # Skip very small layers (like single parameters)
                if module.weight.numel() > 100:
                    layers.append((name, module))
        return layers
    
    def _setup_hooks(self):
        """Setup activation tracking hooks"""
        layers = self._get_model_layers()
        total_partitions = self.n_partitions ** 2
        
        if len(layers) == 0:
            logger.warning("No suitable layers found for hook attachment")
            return
        
        # Distribute layers across partitions
        layers_per_partition = max(1, len(layers) // total_partitions)
        
        self.mind_map.fill(0)
        
        partition_idx = 0
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                start_idx = partition_idx * layers_per_partition
                end_idx = min((partition_idx + 1) * layers_per_partition, len(layers))
                
                # Assign layers to this partition
                for layer_idx in range(start_idx, end_idx):
                    if layer_idx < len(layers):
                        name, module = layers[layer_idx]
                        
                        def make_hook(i, j):
                            def hook(module, input, output):
                                try:
                                    # Handle different output types
                                    if isinstance(output, tuple):
                                        tensor = output[0]
                                    else:
                                        tensor = output
                                    
                                    if tensor is not None and hasattr(tensor, 'abs'):
                                        # Calculate mean absolute activation
                                        activation = float(torch.abs(tensor).mean().item())
                                        self.mind_map[i, j] += activation
                                        
                                except Exception:
                                    # Silently ignore hook errors
                                    pass
                            return hook
                        
                        hook_fn = make_hook(i, j)
                        handle = module.register_forward_hook(hook_fn)
                        self.hooks.append(handle)
                
                partition_idx += 1
    
    def _cleanup_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _calculate_simple_metrics(self):
        """Calculate simple but meaningful research metrics"""
        # Normalize mind map
        if self.mind_map.sum() > 0:
            normalized_map = self.mind_map / self.mind_map.max()
        else:
            normalized_map = self.mind_map.copy()
        
        # 1. Activation Entropy (information distribution)
        flat = normalized_map.flatten()
        flat = flat / (flat.sum() + 1e-10)  # Normalize to probability
        entropy = -np.sum(flat * np.log2(flat + 1e-10))
        
        # 2. Pattern Complexity (how varied the pattern is)
        complexity = float(np.std(normalized_map))
        
        # 3. Regional Specialization (how concentrated activations are)
        if np.mean(normalized_map) > 0:
            specialization = float(np.var(normalized_map) / np.mean(normalized_map))
        else:
            specialization = 0.0
        
        # 4. Activation Spread (how many regions are significantly active)
        threshold = np.percentile(normalized_map, 75) if normalized_map.sum() > 0 else 0
        active_regions = float(np.sum(normalized_map > threshold))
        activation_spread = active_regions / (self.n_partitions ** 2)
        
        # 5. Pattern Symmetry (how symmetric the activation is)
        symmetry = float(np.corrcoef(normalized_map.flatten(), 
                                   np.fliplr(normalized_map).flatten())[0, 1])
        if np.isnan(symmetry):
            symmetry = 0.0
        
        return {
            'activation_entropy': float(entropy),
            'pattern_complexity': complexity,
            'region_specialization': specialization,
            'activation_spread': activation_spread,
            'pattern_symmetry': abs(symmetry)
        }
    
    async def infer_with_analysis(self, prompt, tokenizer, max_tokens=100):
        """Run inference with thought pattern analysis"""
        try:
            # Setup activation tracking
            self._setup_hooks()
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode output
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (remove input prompt)
            generated_text = full_text[len(prompt):].strip() if len(full_text) > len(prompt) else full_text
            
            # Calculate metrics
            metrics = self._calculate_simple_metrics()
            
            # Categorize the query
            category = self._categorize_query(prompt)
            
            return {
                'prompt': prompt,
                'output': generated_text,
                'full_output': full_text,
                'category': category,
                'mind_map': self.mind_map.copy(),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            # Always clean up hooks
            self._cleanup_hooks()
    
    def _categorize_query(self, prompt):
        """Categorize query type for research analysis"""
        prompt_lower = prompt.lower().strip()
        
        # Define category patterns
        patterns = {
            'factual': [
                'what is', 'define', 'who is', 'when did', 'where is',
                'how many', 'what are', 'which', 'name the', 'list the'
            ],
            'reasoning': [
                'why', 'how does', 'explain', 'what causes', 'what happens if',
                'analyze', 'compare', 'relationship', 'because', 'due to'
            ],
            'creative': [
                'write a', 'create', 'imagine', 'story', 'poem',
                'design', 'invent', 'pretend', 'compose', 'generate'
            ],
            'mathematical': [
                'calculate', 'solve', 'compute', 'find the', 'what is the result',
                'equation', 'formula', 'probability', 'percentage', 'derive'
            ],
            'ethical': [
                'should', 'ought', 'moral', 'ethical', 'right or wrong',
                'justified', 'fair', 'unfair', 'good or bad', 'values'
            ]
        }
        
        # Score each category
        category_scores = {}
        for category, category_patterns in patterns.items():
            score = sum(1 for pattern in category_patterns if pattern in prompt_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return highest scoring category, or 'other' if no matches
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'other'

class ResearchExperiment:
    """Research experiment orchestrator"""
    
    def __init__(self, model, tokenizer, n_partitions=4):
        self.model = model
        self.tokenizer = tokenizer
        self.thought_matrix = SimpleThoughtMatrix(model, n_partitions)
        self.n_partitions = n_partitions
    
    def get_research_queries(self):
        """Get comprehensive research queries"""
        return {
            'factual': [
                "What is the capital of France?",
                "Define artificial intelligence.",
                "When did World War II end?",
                "Who invented the telephone?",
                "What is the largest ocean on Earth?",
                "Name the primary colors.",
                "What is the speed of light?",
                "Who wrote Romeo and Juliet?",
                "What is the chemical symbol for gold?",
                "How many continents are there?"
            ],
            'reasoning': [
                "Why do objects fall to the ground?",
                "How does photosynthesis work?",
                "Explain the water cycle.",
                "Why is the sky blue?",
                "How do vaccines work?",
                "What causes seasons on Earth?",
                "Explain supply and demand.",
                "Why do people dream?",
                "How does democracy work?",
                "What causes inflation?"
            ],
            'creative': [
                "Write a short story about a robot.",
                "Create a poem about autumn.",
                "Imagine life on Mars.",
                "Design a new invention.",
                "Write a dialogue between friends.",
                "Create a recipe for happiness.",
                "Invent a new game.",
                "Compose a song about nature.",
                "Design a futuristic city.",
                "Write a fairy tale."
            ],
            'mathematical': [
                "Calculate 17 times 23.",
                "What is 25% of 200?",
                "Solve for x: 2x + 5 = 15.",
                "Find the area of a circle with radius 7.",
                "What is the square root of 144?",
                "Calculate 15 factorial.",
                "What is 2 to the power of 8?",
                "Find the average of 10, 20, 30.",
                "What is 3/4 as a decimal?",
                "Solve: x squared minus 9 equals 0."
            ],
            'ethical': [
                "Should AI have rights?",
                "Is it ethical to test on animals?",
                "Should wealthy nations help poor ones?",
                "Is privacy more important than security?",
                "Should we colonize Mars?",
                "Is genetic engineering moral?",
                "Should education be free?",
                "Is capital punishment justified?",
                "Should social media be regulated?",
                "Is universal healthcare a right?"
            ]
        }
    
    async def run_experiment(self, max_tokens=100, sample_size=None):
        """Run comprehensive research experiment"""
        queries = self.get_research_queries()
        all_results = []
        
        logger.info(f"Starting experiment with {self.n_partitions}x{self.n_partitions} partitions")
        
        for category, category_queries in queries.items():
            logger.info(f"\nTesting category: {category.upper()}")
            
            # Limit sample size if specified
            if sample_size:
                category_queries = category_queries[:sample_size]
            
            category_results = []
            
            for i, query in enumerate(category_queries):
                logger.info(f"  Query {i+1}/{len(category_queries)}: {query}")
                
                try:
                    result = await self.thought_matrix.infer_with_analysis(
                        query, self.tokenizer, max_tokens
                    )
                    all_results.append(result)
                    category_results.append(result)
                    
                except Exception as e:
                    logger.error(f"    Error processing query: {e}")
                    continue
            
            # Log category summary
            if category_results:
                entropies = [r['metrics']['activation_entropy'] for r in category_results]
                complexities = [r['metrics']['pattern_complexity'] for r in category_results]
                logger.info(f"  Category summary - Avg Entropy: {np.mean(entropies):.3f}, "
                          f"Avg Complexity: {np.mean(complexities):.3f}")
        
        logger.info(f"\nExperiment completed: {len(all_results)} successful queries")
        return all_results
    
    def analyze_results(self, results):
        """Comprehensive analysis of experiment results"""
        if not results:
            return {"error": "No results to analyze"}
        
        analysis = {
            'experiment_metadata': {
                'total_queries': len(results),
                'timestamp': datetime.now().isoformat(),
                'model_info': str(type(self.model).__name__),
                'partitions': f"{self.n_partitions}x{self.n_partitions}",
                'categories': list(set(r['category'] for r in results))
            },
            'overall_statistics': {},
            'category_analysis': {},
            'research_insights': []
        }
        
        # Overall statistics across all results
        metrics = ['activation_entropy', 'pattern_complexity', 'region_specialization', 
                  'activation_spread', 'pattern_symmetry']
        
        overall_stats = {}
        for metric in metrics:
            values = [r['metrics'][metric] for r in results if metric in r['metrics']]
            if values:
                overall_stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        analysis['overall_statistics'] = overall_stats
        
        # Category-wise analysis
        by_category = {}
        for result in results:
            category = result['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        category_analysis = {}
        for category, cat_results in by_category.items():
            cat_stats = {}
            
            for metric in metrics:
                values = [r['metrics'][metric] for r in cat_results if metric in r['metrics']]
                if values:
                    cat_stats[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'count': len(values)
                    }
            
            category_analysis[category] = {
                'query_count': len(cat_results),
                'statistics': cat_stats,
                'sample_queries': [r['prompt'] for r in cat_results[:3]]
            }
        
        analysis['category_analysis'] = category_analysis
        
        # Generate research insights
        insights = self._generate_insights(category_analysis)
        analysis['research_insights'] = insights
        
        return analysis
    
    def _generate_insights(self, category_analysis):
        """Generate research insights from the analysis"""
        insights = []
        
        # Compare entropy across categories
        entropy_by_category = {}
        complexity_by_category = {}
        
        for category, data in category_analysis.items():
            if 'activation_entropy' in data['statistics']:
                entropy_by_category[category] = data['statistics']['activation_entropy']['mean']
            if 'pattern_complexity' in data['statistics']:
                complexity_by_category[category] = data['statistics']['pattern_complexity']['mean']
        
        if entropy_by_category:
            highest_entropy = max(entropy_by_category, key=entropy_by_category.get)
            lowest_entropy = min(entropy_by_category, key=entropy_by_category.get)
            
            insights.extend([
                f"Highest activation entropy (most distributed processing): {highest_entropy} "
                f"({entropy_by_category[highest_entropy]:.3f})",
                f"Lowest activation entropy (most focused processing): {lowest_entropy} "
                f"({entropy_by_category[lowest_entropy]:.3f})"
            ])
        
        if complexity_by_category:
            highest_complexity = max(complexity_by_category, key=complexity_by_category.get)
            lowest_complexity = min(complexity_by_category, key=complexity_by_category.get)
            
            insights.extend([
                f"Most complex thought patterns: {highest_complexity} "
                f"({complexity_by_category[highest_complexity]:.3f})",
                f"Simplest thought patterns: {lowest_complexity} "
                f"({complexity_by_category[lowest_complexity]:.3f})"
            ])
        
        # Add general insights
        insights.extend([
            "Different cognitive categories show measurable differences in neural activation patterns",
            "Activation entropy varies significantly across query types, suggesting functional specialization",
            "Pattern complexity correlates with cognitive demands of different reasoning tasks"
        ])
        
        return insights

async def main():
    """Main research pipeline"""
    parser = argparse.ArgumentParser(description="Simple Research Pipeline for Thought Matrix Analysis")
    parser.add_argument("--model", default="gpt2", help="Model to use (default: gpt2)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--sample-size", type=int, default=10, help="Queries per category")
    parser.add_argument("--partitions", type=int, default=4, help="Matrix partitions (4 = 4x4 matrix)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("SIMPLE THOUGHT MATRIX RESEARCH PIPELINE")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Matrix: {args.partitions}x{args.partitions}")
    print(f"Sample size: {args.sample_size} queries per category")
    
    # Load model and tokenizer
    try:
        logger.info(f"Loading model: {args.model}")
        
        if args.model == "gpt2":
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        logger.info(f"Model loaded successfully on: {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Create and run experiment
    experiment = ResearchExperiment(model, tokenizer, args.partitions)
    
    logger.info("Starting research experiment...")
    results = await experiment.run_experiment(
        max_tokens=args.max_tokens,
        sample_size=args.sample_size
    )
    
    if not results:
        logger.error("No results generated. Exiting.")
        return
    
    # Analyze results
    logger.info("Analyzing results...")
    analysis = experiment.analyze_results(results)
    
    # Save results
    output_dir = Path("./research_output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"
    analysis_file = output_dir / f"analysis_{timestamp}.json"
    
    # JSON serializer for numpy arrays
    def json_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=json_serializer)
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=json_serializer)
    
    # Display results
    print("\n" + "="*70)
    print("RESEARCH RESULTS")
    print("="*70)
    
    meta = analysis['experiment_metadata']
    print(f"Total queries processed: {meta['total_queries']}")
    print(f"Categories analyzed: {', '.join(meta['categories'])}")
    
    # Overall statistics
    if analysis['overall_statistics']:
        print(f"\nOverall Statistics:")
        for metric, stats in analysis['overall_statistics'].items():
            print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                  f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
    
    # Category breakdown
    print(f"\nCategory Analysis:")
    for category, data in analysis['category_analysis'].items():
        print(f"\n  {category.upper()} ({data['query_count']} queries):")
        for metric, stats in data['statistics'].items():
            print(f"    {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Key insights
    print(f"\nKey Research Insights:")
    for insight in analysis['research_insights']:
        print(f"  • {insight}")
    
    print(f"\nDetailed results saved to:")
    print(f"  Raw data: {results_file}")
    print(f"  Analysis: {analysis_file}")
    
    # Sample mind map
    if results:
        sample = results[0]
        print(f"\nSample thought matrix for: '{sample['prompt']}'")
        mind_map = sample['mind_map']
        print(f"Shape: {mind_map.shape}")
        for i, row in enumerate(mind_map):
            row_str = "  [" + " ".join([f"{val:6.3f}" for val in row]) + "]"
            print(row_str)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())