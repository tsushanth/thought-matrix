#!/usr/bin/env python3
"""
Basic Research Pipeline - Works with minimal dependencies
========================================================

This version works with just transformers + torch and provides
a solid foundation for your research paper.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicThoughtMatrix:
    """Enhanced version of the basic thought matrix"""
    
    def __init__(self, model, n_partitions=4):
        self.model = model
        self.n_partitions = n_partitions
        self.device = next(model.parameters()).device
        self.mind_map = np.zeros((n_partitions, n_partitions))
        self.activation_variance = np.zeros((n_partitions, n_partitions))
        self.hooks = []
        
    def _get_functional_layers(self):
        """Get layers grouped by function"""
        attention_layers = []
        mlp_layers = []
        embedding_layers = []
        other_layers = []
        
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'weight'):
                attention_layers.append((name, module))
            elif any(x in name.lower() for x in ['mlp', 'ffn', 'feed_forward']) and hasattr(module, 'weight'):
                mlp_layers.append((name, module))
            elif 'embed' in name.lower() and hasattr(module, 'weight'):
                embedding_layers.append((name, module))
            elif hasattr(module, 'weight') and len(list(module.parameters())) > 0:
                other_layers.append((name, module))
        
        return {
            'attention': attention_layers,
            'mlp': mlp_layers,
            'embedding': embedding_layers,
            'other': other_layers
        }
    
    def _setup_enhanced_hooks(self):
        """Setup hooks with functional awareness"""
        functional_layers = self._get_functional_layers()
        all_layers = []
        
        # Collect all layers with their functions
        for func_type, layers in functional_layers.items():
            for name, module in layers:
                all_layers.append((name, module, func_type))
        
        total_partitions = self.n_partitions ** 2
        
        if len(all_layers) < total_partitions:
            # Repeat layers if needed
            all_layers = (all_layers * ((total_partitions // len(all_layers)) + 1))[:total_partitions]
        
        # Distribute across partitions
        layers_per_partition = max(1, len(all_layers) // total_partitions)
        
        self.mind_map.fill(0)
        self.activation_variance.fill(0)
        
        idx = 0
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                start_idx = idx * layers_per_partition
                end_idx = min((idx + 1) * layers_per_partition, len(all_layers))
                
                partition_activations = []
                
                for layer_idx in range(start_idx, end_idx):
                    if layer_idx < len(all_layers):
                        name, module, func_type = all_layers[layer_idx]
                        
                        def make_hook(i, j, func_type):
                            def hook(module, input, output):
                                try:
                                    if isinstance(output, tuple):
                                        output = output[0]
                                    
                                    # Calculate mean and variance
                                    activation_tensor = torch.abs(output)
                                    mean_activation = activation_tensor.mean().item()
                                    var_activation = activation_tensor.var().item()
                                    
                                    self.mind_map[i, j] += mean_activation
                                    self.activation_variance[i, j] += var_activation
                                    
                                except Exception as e:
                                    pass  # Ignore hook errors
                            return hook
                        
                        hook_fn = make_hook(i, j, func_type)
                        handle = module.register_forward_hook(hook_fn)
                        self.hooks.append(handle)
                
                idx += 1
    
    def _cleanup_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _calculate_metrics(self):
        """Calculate research metrics"""
        # Normalize mind map
        if self.mind_map.sum() > 0:
            normalized_map = self.mind_map / self.mind_map.max()
        else:
            normalized_map = self.mind_map.copy()
        
        # Calculate entropy
        flat = normalized_map.flatten()
        flat = flat / (flat.sum() + 1e-10)
        entropy = -np.sum(flat * np.log2(flat + 1e-10))
        
        # Calculate specialization (how concentrated activations are)
        specialization = np.var(normalized_map) / (np.mean(normalized_map) + 1e-10)
        
        # Calculate complexity (pattern variation)
        complexity = np.std(normalized_map)
        
        # Calculate connectivity (correlation between adjacent regions)
        connectivity = 0.0
        if normalized_map.shape[0] > 1:
            correlations = []
            for i in range(normalized_map.shape[0] - 1):
                for j in range(normalized_map.shape[1] - 1):
                    # Horizontal and vertical connectivity
                    if j + 1 < normalized_map.shape[1]:
                        corr = np.corrcoef(normalized_map[i, :j+2])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    if i + 1 < normalized_map.shape[0]:
                        corr = np.corrcoef(normalized_map[:i+2, j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            if correlations:
                connectivity = np.mean(correlations)
        
        return {
            'activation_entropy': entropy,
            'region_specialization': specialization,
            'pattern_complexity': complexity,
            'cross_region_connectivity': connectivity
        }
    
    async def infer_with_analysis(self, prompt, tokenizer, max_tokens=100):
        """Run inference with comprehensive analysis"""
        try:
            # Setup hooks
            self._setup_enhanced_hooks()
            
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
            
            # Decode output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            # Categorize query
            category = self._categorize_query(prompt)
            
            return {
                'prompt': prompt,
                'output': output_text,
                'category': category,
                'mind_map': self.mind_map.copy(),
                'activation_variance': self.activation_variance.copy(),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            self._cleanup_hooks()
    
    def _categorize_query(self, prompt):
        """Enhanced query categorization"""
        prompt_lower = prompt.lower()
        
        # Factual patterns
        factual_patterns = ['what is', 'define', 'who is', 'when did', 'where is', 
                           'how many', 'what are', 'which', 'name the']
        
        # Reasoning patterns  
        reasoning_patterns = ['why', 'how does', 'explain', 'what causes', 
                             'what happens if', 'analyze', 'compare', 'relationship']
        
        # Creative patterns
        creative_patterns = ['write a', 'create', 'imagine', 'story', 'poem', 
                            'design', 'invent', 'pretend']
        
        # Mathematical patterns
        math_patterns = ['calculate', 'solve', 'what is.*\\+', 'what is.*\\-', 
                        'find the', 'prove', 'derivative', 'integral']
        
        # Ethical patterns
        ethical_patterns = ['should', 'is it right', 'moral', 'ethical', 
                           'ought to', 'fair', 'just']
        
        # Count matches for each category
        scores = {
            'factual': sum(1 for pattern in factual_patterns if pattern in prompt_lower),
            'reasoning': sum(1 for pattern in reasoning_patterns if pattern in prompt_lower),
            'creative': sum(1 for pattern in creative_patterns if pattern in prompt_lower),
            'mathematical': sum(1 for pattern in math_patterns if pattern in prompt_lower),
            'ethical': sum(1 for pattern in ethical_patterns if pattern in prompt_lower)
        }
        
        # Return category with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return 'other'

class BasicResearchExperiment:
    """Research experiment using basic thought matrix"""
    
    def __init__(self, model, tokenizer, n_partitions=4):
        self.model = model
        self.tokenizer = tokenizer
        self.thought_matrix = BasicThoughtMatrix(model, n_partitions)
        self.results = []
    
    def get_comprehensive_test_queries(self):
        """Get comprehensive test queries for research"""
        return {
            'factual': [
                "What is the capital of France?",
                "Define artificial intelligence.",
                "When did World War II end?", 
                "Who invented the telephone?",
                "How many planets are in our solar system?",
                "What is the largest ocean on Earth?",
                "Name the primary colors.",
                "What is the speed of light?",
                "Define photosynthesis.",
                "Who wrote Romeo and Juliet?",
                "What is the chemical symbol for gold?",
                "When was the United Nations founded?",
                "What is the tallest mountain in the world?",
                "Who painted the Mona Lisa?",
                "What is the currency of Japan?"
            ],
            'reasoning': [
                "Why do objects fall to the ground?",
                "How does the greenhouse effect work?", 
                "Explain the relationship between supply and demand.",
                "What causes seasons on Earth?",
                "Why is the sky blue?",
                "How do vaccines work?",
                "What happens if you heat ice?",
                "Analyze the causes of inflation.",
                "Why do people dream?",
                "How does democracy differ from autocracy?",
                "Explain why some materials conduct electricity.",
                "What causes ocean tides?",
                "How does natural selection work?",
                "Why do different languages exist?",
                "Explain the water cycle."
            ],
            'creative': [
                "Write a short story about a time traveler.",
                "Create a poem about autumn.",
                "Imagine life on another planet.",
                "Design a new type of transportation.",
                "Write a dialogue between two robots.",
                "Create a recipe for happiness.",
                "Invent a new sport.",
                "Write a letter from the future.",
                "Create a superhero origin story.", 
                "Design a perfect city.",
                "Write a song about friendship.",
                "Imagine a world without gravity.",
                "Create a new holiday tradition.",
                "Design an alien species.",
                "Write a fairy tale with a modern twist."
            ],
            'mathematical': [
                "Calculate 17 × 23.",
                "What is 25% of 200?",
                "Solve for x: 2x + 5 = 15.",
                "Find the area of a circle with radius 7.",
                "What is the square root of 144?",
                "Calculate compound interest on $1000 at 5% for 3 years.",
                "Find the derivative of x².",
                "What is 15! (15 factorial)?",
                "Solve: x² - 5x + 6 = 0.",
                "Calculate the probability of rolling a 6.",
                "What is 2³ + 3² - 4¹?",
                "Find the slope of y = 3x + 2.",
                "Calculate the volume of a sphere with radius 5.",
                "What is the sum of angles in a triangle?",
                "Solve: 3x + 2y = 12, x - y = 1."
            ],
            'ethical': [
                "Should artificial intelligence have rights?",
                "Is it ethical to test on animals?",
                "What are the moral implications of genetic engineering?",
                "Should wealthy nations help poor ones?",
                "Is capital punishment justified?",
                "What are the ethics of privacy vs security?",
                "Should we colonize Mars?",
                "Is it fair to have different wages?",
                "What are the ethics of autonomous weapons?",
                "Should we extend human lifespan indefinitely?",
                "Is it right to edit human genes?",
                "Should social media be regulated?",
                "What are the ethics of AI in healthcare?",
                "Is universal basic income moral?",
                "Should we preserve endangered languages?"
            ]
        }
    
    async def run_comprehensive_experiment(self, max_tokens=150, sample_size=None):
        """Run comprehensive research experiment"""
        test_queries = self.get_comprehensive_test_queries()
        results = []
        
        for category, queries in test_queries.items():
            logger.info(f"Testing category: {category}")
            
            # Limit sample size if specified
            if sample_size:
                queries = queries[:sample_size]
            
            category_results = []
            
            for i, query in enumerate(queries):
                logger.info(f"  Query {i+1}/{len(queries)}: {query}")
                
                result = await self.thought_matrix.infer_with_analysis(
                    query, self.tokenizer, max_tokens
                )
                
                results.append(result)
                category_results.append(result)
            
            # Log category summary
            if category_results:
                avg_entropy = np.mean([r['metrics']['activation_entropy'] for r in category_results])
                avg_complexity = np.mean([r['metrics']['pattern_complexity'] for r in category_results])
                logger.info(f"  {category} - Avg Entropy: {avg_entropy:.3f}, Avg Complexity: {avg_complexity:.3f}")
        
        return results
    
    def analyze_results(self, results):
        """Analyze experiment results"""
        analysis = {
            'metadata': {
                'total_queries': len(results),
                'timestamp': datetime.now().isoformat(),
                'categories': list(set(r['category'] for r in results))
            },
            'overall_statistics': {},
            'category_analysis': {},
            'statistical_insights': []
        }
        
        # Overall statistics
        all_entropy = [r['metrics']['activation_entropy'] for r in results]
        all_complexity = [r['metrics']['pattern_complexity'] for r in results]
        all_specialization = [r['metrics']['region_specialization'] for r in results]
        all_connectivity = [r['metrics']['cross_region_connectivity'] for r in results]
        
        analysis['overall_statistics'] = {
            'entropy': {
                'mean': float(np.mean(all_entropy)),
                'std': float(np.std(all_entropy)),
                'min': float(np.min(all_entropy)),
                'max': float(np.max(all_entropy))
            },
            'complexity': {
                'mean': float(np.mean(all_complexity)),
                'std': float(np.std(all_complexity)),
                'min': float(np.min(all_complexity)),
                'max': float(np.max(all_complexity))
            },
            'specialization': {
                'mean': float(np.mean(all_specialization)),
                'std': float(np.std(all_specialization))
            },
            'connectivity': {
                'mean': float(np.mean(all_connectivity)),
                'std': float(np.std(all_connectivity))
            }
        }
        
        # Category analysis
        by_category = {}
        for result in results:
            cat = result['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)
        
        for category, cat_results in by_category.items():
            cat_entropy = [r['metrics']['activation_entropy'] for r in cat_results]
            cat_complexity = [r['metrics']['pattern_complexity'] for r in cat_results]
            cat_specialization = [r['metrics']['region_specialization'] for r in cat_results]
            
            analysis['category_analysis'][category] = {
                'count': len(cat_results),
                'entropy': {
                    'mean': float(np.mean(cat_entropy)),
                    'std': float(np.std(cat_entropy))
                },
                'complexity': {
                    'mean': float(np.mean(cat_complexity)),
                    'std': float(np.std(cat_complexity))
                },
                'specialization': {
                    'mean': float(np.mean(cat_specialization)),
                    'std': float(np.std(cat_specialization))
                },
                'sample_queries': [r['prompt'] for r in cat_results[:3]]
            }
        
        # Generate insights
        category_stats = analysis['category_analysis']
        
        # Find extremes
        entropy_by_cat = {cat: stats['entropy']['mean'] for cat, stats in category_stats.items()}
        complexity_by_cat = {cat: stats['complexity']['mean'] for cat, stats in category_stats.items()}
        
        if entropy_by_cat:
            highest_entropy_cat = max(entropy_by_cat, key=entropy_by_cat.get)
            lowest_entropy_cat = min(entropy_by_cat, key=entropy_by_cat.get)
            
            analysis['statistical_insights'].extend([
                f"Highest activation entropy: {highest_entropy_cat} ({entropy_by_cat[highest_entropy_cat]:.3f})",
                f"Lowest activation entropy: {lowest_entropy_cat} ({entropy_by_cat[lowest_entropy_cat]:.3f})"
            ])
        
        if complexity_by_cat:
            highest_complexity_cat = max(complexity_by_cat, key=complexity_by_cat.get)
            lowest_complexity_cat = min(complexity_by_cat, key=complexity_by_cat.get)
            
            analysis['statistical_insights'].extend([
                f"Highest pattern complexity: {highest_complexity_cat} ({complexity_by_cat[highest_complexity_cat]:.3f})",
                f"Lowest pattern complexity: {lowest_complexity_cat} ({complexity_by_cat[lowest_complexity_cat]:.3f})"
            ])
        
        return analysis

async def main():
    """Main research pipeline"""
    parser = argparse.ArgumentParser(description="Basic Research Pipeline")
    parser.add_argument("--model", default="gpt2", help="Model to use")
    parser.add_argument("--experiment", choices=["basic", "focused"], default="basic")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--sample-size", type=int, help="Sample size per category")
    parser.add_argument("--n-partitions", type=int, default=4, help="Number of partitions")
    
    args = parser.parse_args()
    
    print("="*60)
    print("BASIC THOUGHT MATRIX RESEARCH PIPELINE")
    print("="*60)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    
    try:
        if args.model == "gpt2":
            # Use GPT-2 (no token required)
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            # Use other models (may require HuggingFace token)
            model = AutoModelForCausalLM.from_pretrained(args.model)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        logger.info(f"Model loaded on: {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Create experiment
    experiment = BasicResearchExperiment(model, tokenizer, args.n_partitions)
    
    # Run experiment
    logger.info("Starting comprehensive experiment...")
    results = await experiment.run_comprehensive_experiment(
        max_tokens=args.max_tokens,
        sample_size=args.sample_size
    )
    
    logger.info(f"Completed experiment with {len(results)} queries")
    
    # Analyze results
    logger.info("Analyzing results...")
    analysis = experiment.analyze_results(results)
    
    # Save results
    output_dir = Path("./research_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"experiment_results_{timestamp}.json"
    analysis_file = output_dir / f"experiment_analysis_{timestamp}.json"
    
    # Save with proper JSON serialization
    def json_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=json_serializer)
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=json_serializer)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    
    print(f"Total queries: {analysis['metadata']['total_queries']}")
    print(f"Categories: {', '.join(analysis['metadata']['categories'])}")
    
    print(f"\nOverall Statistics:")
    overall = analysis['overall_statistics']
    print(f"  Entropy: {overall['entropy']['mean']:.3f} ± {overall['entropy']['std']:.3f}")
    print(f"  Complexity: {overall['complexity']['mean']:.3f} ± {overall['complexity']['std']:.3f}")
    print(f"  Specialization: {overall['specialization']['mean']:.3f} ± {overall['specialization']['std']:.3f}")
    
    print(f"\nKey Insights:")
    for insight in analysis['statistical_insights']:
        print(f"  • {insight}")
    
    print(f"\nBy Category:")
    for category, stats in analysis['category_analysis'].items():
        print(f"  {category.upper()}: {stats['count']} queries")
        print(f"    Entropy: {stats['entropy']['mean']:.3f}")
        print(f"    Complexity: {stats['complexity']['mean']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  Raw results: {results_file}")
    print(f"  Analysis: {analysis_file}")

if __name__ == "__main__":
    asyncio.run(main())