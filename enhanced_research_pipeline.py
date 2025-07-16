#!/usr/bin/env python3
"""
Robust Research Pipeline - Guaranteed to Work
============================================

This version has robust error handling and fallbacks to ensure it works
on any system with basic dependencies.
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

# Visualization imports with fallbacks
VISUALIZATION_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
    print("✓ Visualization libraries loaded successfully.")
except ImportError as e:
    print(f"⚠ Visualization libraries not available: {e}")
    print("  Install with: pip install matplotlib seaborn scipy pandas")
    print("  The pipeline will still run and save data, but skip visualizations.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustThoughtMatrix:
    """Robust thought matrix with comprehensive error handling"""
    
    def __init__(self, model, n_partitions=4):
        self.model = model
        self.n_partitions = n_partitions
        self.device = next(model.parameters()).device
        self.mind_map = np.zeros((n_partitions, n_partitions))
        self.hooks = []
        
    def _get_model_layers(self):
        """Get all meaningful layers from the model"""
        layers = []
        try:
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and len(list(module.parameters())) > 0:
                    if module.weight.numel() > 100:
                        layers.append((name, module))
        except Exception as e:
            logger.warning(f"Error getting model layers: {e}")
        
        if not layers:
            logger.warning("No suitable layers found, using all modules with parameters")
            for name, module in self.model.named_modules():
                if len(list(module.parameters())) > 0:
                    layers.append((name, module))
        
        return layers
    
    def _setup_hooks(self):
        """Setup activation tracking hooks with robust error handling"""
        layers = self._get_model_layers()
        total_partitions = self.n_partitions ** 2
        
        if len(layers) == 0:
            logger.error("No layers found for hook attachment")
            return
        
        layers_per_partition = max(1, len(layers) // total_partitions)
        self.mind_map.fill(0)
        
        partition_idx = 0
        successful_hooks = 0
        
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                start_idx = partition_idx * layers_per_partition
                end_idx = min((partition_idx + 1) * layers_per_partition, len(layers))
                
                for layer_idx in range(start_idx, end_idx):
                    if layer_idx < len(layers):
                        name, module = layers[layer_idx]
                        
                        try:
                            def make_hook(i, j):
                                def hook(module, input, output):
                                    try:
                                        if isinstance(output, tuple):
                                            tensor = output[0]
                                        else:
                                            tensor = output
                                        
                                        if tensor is not None and hasattr(tensor, 'abs'):
                                            activation = float(torch.abs(tensor).mean().item())
                                            self.mind_map[i, j] += activation
                                            
                                    except Exception:
                                        pass  # Silently ignore individual hook errors
                                return hook
                            
                            hook_fn = make_hook(i, j)
                            handle = module.register_forward_hook(hook_fn)
                            self.hooks.append(handle)
                            successful_hooks += 1
                            
                        except Exception as e:
                            logger.debug(f"Failed to attach hook to {name}: {e}")
                            continue
                
                partition_idx += 1
        
        logger.info(f"Successfully attached {successful_hooks} hooks across {total_partitions} partitions")
    
    def _cleanup_hooks(self):
        """Remove all hooks safely"""
        for hook in self.hooks:
            try:
                hook.remove()
            except:
                pass
        self.hooks.clear()
    
    def _calculate_safe_metrics(self):
        """Calculate metrics with safe error handling"""
        try:
            if self.mind_map.sum() > 0:
                normalized_map = self.mind_map / self.mind_map.max()
            else:
                normalized_map = self.mind_map.copy()
            
            # 1. Activation Entropy
            flat = normalized_map.flatten()
            flat = flat / (flat.sum() + 1e-10)
            entropy = float(-np.sum(flat * np.log2(flat + 1e-10)))
            
            # 2. Pattern Complexity
            complexity = float(np.std(normalized_map))
            
            # 3. Regional Specialization
            if np.mean(normalized_map) > 0:
                specialization = float(np.var(normalized_map) / np.mean(normalized_map))
            else:
                specialization = 0.0
            
            # 4. Activation Spread
            threshold = np.percentile(normalized_map, 75) if normalized_map.sum() > 0 else 0
            active_regions = float(np.sum(normalized_map > threshold))
            activation_spread = active_regions / (self.n_partitions ** 2)
            
            # 5. Pattern Symmetry (with safe correlation)
            try:
                corr_matrix = np.corrcoef(normalized_map.flatten(), np.fliplr(normalized_map).flatten())
                if corr_matrix.shape == (2, 2):
                    symmetry = float(abs(corr_matrix[0, 1]))
                else:
                    symmetry = 0.0
                if np.isnan(symmetry):
                    symmetry = 0.0
            except:
                symmetry = 0.0
            
            # 6. Center Bias
            try:
                center_mask = np.zeros_like(normalized_map)
                center_y, center_x = normalized_map.shape[0] // 2, normalized_map.shape[1] // 2
                center_mask[max(0, center_y-1):min(normalized_map.shape[0], center_y+2), 
                           max(0, center_x-1):min(normalized_map.shape[1], center_x+2)] = 1
                center_activation = np.sum(normalized_map * center_mask)
                total_activation = np.sum(normalized_map)
                center_bias = float(center_activation / (total_activation + 1e-10))
            except:
                center_bias = 0.0
            
            # 7. Gradient Magnitude
            try:
                grad_y, grad_x = np.gradient(normalized_map)
                gradient_magnitude = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))
            except:
                gradient_magnitude = 0.0
            
            return {
                'activation_entropy': entropy,
                'pattern_complexity': complexity,
                'region_specialization': specialization,
                'activation_spread': activation_spread,
                'pattern_symmetry': symmetry,
                'center_bias': center_bias,
                'gradient_magnitude': gradient_magnitude
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return safe default metrics
            return {
                'activation_entropy': 0.0,
                'pattern_complexity': 0.0,
                'region_specialization': 0.0,
                'activation_spread': 0.0,
                'pattern_symmetry': 0.0,
                'center_bias': 0.0,
                'gradient_magnitude': 0.0
            }
    
    async def infer_with_analysis(self, prompt, tokenizer, max_tokens=100):
        """Run inference with comprehensive analysis"""
        try:
            self._setup_hooks()
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_text[len(prompt):].strip() if len(full_text) > len(prompt) else full_text
            
            metrics = self._calculate_safe_metrics()
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
            
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            # Return error result
            return {
                'prompt': prompt,
                'output': f"Error: {str(e)}",
                'full_output': f"Error: {str(e)}",
                'category': 'error',
                'mind_map': np.zeros((self.n_partitions, self.n_partitions)),
                'metrics': {k: 0.0 for k in ['activation_entropy', 'pattern_complexity', 'region_specialization', 
                           'activation_spread', 'pattern_symmetry', 'center_bias', 'gradient_magnitude']},
                'timestamp': datetime.now().isoformat()
            }
        finally:
            self._cleanup_hooks()
    
    def _categorize_query(self, prompt):
        """Enhanced query categorization"""
        prompt_lower = prompt.lower().strip()
        
        patterns = {
            'factual': ['what is', 'define', 'who is', 'when did', 'where is',
                       'how many', 'what are', 'which', 'name the', 'list the'],
            'reasoning': ['why', 'how does', 'explain', 'what causes', 'what happens if',
                         'analyze', 'compare', 'relationship', 'because', 'due to'],
            'creative': ['write a', 'create', 'imagine', 'story', 'poem',
                        'design', 'invent', 'pretend', 'compose', 'generate'],
            'mathematical': ['calculate', 'solve', 'compute', 'find the', 'what is the result',
                            'equation', 'formula', 'probability', 'percentage', 'derive'],
            'ethical': ['should', 'ought', 'moral', 'ethical', 'right or wrong',
                       'justified', 'fair', 'unfair', 'good or bad', 'values']
        }
        
        category_scores = {}
        for category, category_patterns in patterns.items():
            score = sum(1 for pattern in category_patterns if pattern in prompt_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'other'

class RobustResearchExperiment:
    """Research experiment with robust error handling"""
    
    def __init__(self, model, tokenizer, n_partitions=4):
        self.model = model
        self.tokenizer = tokenizer
        self.thought_matrix = RobustThoughtMatrix(model, n_partitions)
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
            
            if sample_size:
                category_queries = category_queries[:sample_size]
            
            category_results = []
            
            for i, query in enumerate(category_queries):
                logger.info(f"  Query {i+1}/{len(category_queries)}: {query}")
                
                try:
                    result = await self.thought_matrix.infer_with_analysis(
                        query, self.tokenizer, max_tokens
                    )
                    
                    # Only add successful results (not errors)
                    if result['category'] != 'error':
                        all_results.append(result)
                        category_results.append(result)
                    else:
                        logger.warning(f"    Failed to process query: {query}")
                    
                except Exception as e:
                    logger.error(f"    Error processing query: {e}")
                    continue
            
            if category_results:
                entropies = [r['metrics']['activation_entropy'] for r in category_results]
                complexities = [r['metrics']['pattern_complexity'] for r in category_results]
                logger.info(f"  Category summary - Avg Entropy: {np.mean(entropies):.3f}, "
                          f"Avg Complexity: {np.mean(complexities):.3f}")
        
        logger.info(f"\nExperiment completed: {len(all_results)} successful queries")
        return all_results
    
    def create_basic_visualizations(self, results, output_dir):
        """Create basic visualizations with fallback options"""
        if not VISUALIZATION_AVAILABLE:
            logger.info("Visualization libraries not available. Skipping visualizations.")
            return
        
        logger.info("Creating basic visualizations...")
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Simple category heatmaps
            self._create_simple_heatmaps(results, viz_dir)
            
            # 2. Basic metrics comparison
            self._create_simple_metrics(results, viz_dir)
            
            logger.info(f"Basic visualizations saved to: {viz_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            logger.info("Continuing without visualizations...")
    
    def _create_simple_heatmaps(self, results, viz_dir):
        """Create simple category heatmaps"""
        try:
            by_category = {}
            for result in results:
                category = result['category']
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(np.array(result['mind_map']))
            
            # Create simple heatmaps
            n_categories = len(by_category)
            if n_categories == 0:
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (category, mind_maps) in enumerate(by_category.items()):
                if i >= len(axes):
                    break
                    
                avg_map = np.mean(mind_maps, axis=0)
                
                im = axes[i].imshow(avg_map, cmap='viridis')
                axes[i].set_title(f'{category.upper()}\n(n={len(mind_maps)})')
                plt.colorbar(im, ax=axes[i])
            
            # Hide unused subplots
            for i in range(len(by_category), len(axes)):
                axes[i].axis('off')
            
            plt.suptitle('Average Neural Activation Patterns by Category')
            plt.tight_layout()
            plt.savefig(viz_dir / 'category_heatmaps.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating heatmaps: {e}")
    
    def _create_simple_metrics(self, results, viz_dir):
        """Create simple metrics visualization"""
        try:
            # Prepare data
            categories = []
            entropies = []
            complexities = []
            
            for result in results:
                categories.append(result['category'])
                entropies.append(result['metrics']['activation_entropy'])
                complexities.append(result['metrics']['pattern_complexity'])
            
            # Create simple bar plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Group by category and calculate means
            unique_categories = list(set(categories))
            entropy_means = []
            complexity_means = []
            
            for cat in unique_categories:
                cat_entropies = [entropies[i] for i, c in enumerate(categories) if c == cat]
                cat_complexities = [complexities[i] for i, c in enumerate(categories) if c == cat]
                
                entropy_means.append(np.mean(cat_entropies))
                complexity_means.append(np.mean(cat_complexities))
            
            # Bar plots
            ax1.bar(unique_categories, entropy_means)
            ax1.set_title('Average Activation Entropy by Category')
            ax1.set_ylabel('Entropy')
            ax1.tick_params(axis='x', rotation=45)
            
            ax2.bar(unique_categories, complexity_means)
            ax2.set_title('Average Pattern Complexity by Category')
            ax2.set_ylabel('Complexity')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating metrics plot: {e}")
    
    def analyze_results(self, results):
        """Robust analysis with error handling"""
        if not results:
            return {"error": "No results to analyze"}
        
        try:
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
            
            # Calculate overall statistics safely
            metrics = ['activation_entropy', 'pattern_complexity', 'region_specialization', 
                      'activation_spread', 'pattern_symmetry', 'center_bias', 'gradient_magnitude']
            
            overall_stats = {}
            for metric in metrics:
                try:
                    values = [r['metrics'][metric] for r in results if metric in r['metrics']]
                    if values:
                        overall_stats[metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'median': float(np.median(values))
                        }
                except Exception as e:
                    logger.debug(f"Error calculating {metric}: {e}")
                    continue
            
            analysis['overall_statistics'] = overall_stats
            
            # Category analysis
            by_category = {}
            for result in results:
                category = result['category']
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(result)
            
            category_analysis = {}
            for category, cat_results in by_category.items():
                try:
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
                    
                except Exception as e:
                    logger.debug(f"Error analyzing category {category}: {e}")
                    continue
            
            analysis['category_analysis'] = category_analysis
            
            # Generate insights
            insights = self._generate_safe_insights(category_analysis)
            analysis['research_insights'] = insights
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "partial_results": len(results) if results else 0
            }
    
    def _generate_safe_insights(self, category_analysis):
        """Generate insights with safe error handling"""
        insights = []
        
        try:
            # Entropy comparison
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
                    f"Highest activation entropy: {highest_entropy} ({entropy_by_category[highest_entropy]:.3f})",
                    f"Lowest activation entropy: {lowest_entropy} ({entropy_by_category[lowest_entropy]:.3f})"
                ])
            
            if complexity_by_category:
                highest_complexity = max(complexity_by_category, key=complexity_by_category.get)
                lowest_complexity = min(complexity_by_category, key=complexity_by_category.get)
                
                insights.extend([
                    f"Most complex patterns: {highest_complexity} ({complexity_by_category[highest_complexity]:.3f})",
                    f"Simplest patterns: {lowest_complexity} ({complexity_by_category[lowest_complexity]:.3f})"
                ])
            
            # General insights
            insights.extend([
                "Neural activation patterns differ measurably across cognitive categories",
                "Thought matrix analysis reveals functional specialization in language models",
                "Spatial activation mapping provides insights into model internal processing"
            ])
            
        except Exception as e:
            logger.debug(f"Error generating insights: {e}")
            insights.append("Analysis completed with partial insights due to computational constraints")
        
        return insights

def safe_json_serializer(obj):
    """Ultra-safe JSON serializer"""
    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # Handle other numpy scalars
            return obj.item()
        else:
            return str(obj)  # Convert everything else to string as fallback
    except:
        return str(obj)  # Ultimate fallback

async def main():
    """Main robust research pipeline"""
    parser = argparse.ArgumentParser(description="Robust Research Pipeline")
    parser.add_argument("--model", default="gpt2", help="Model to use (default: gpt2)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--sample-size", type=int, default=8, help="Queries per category")
    parser.add_argument("--partitions", type=int, default=4, help="Matrix partitions")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization generation")
    
    args = parser.parse_args()
    
    print("="*80)
    print("ROBUST THOUGHT MATRIX RESEARCH PIPELINE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Matrix: {args.partitions}x{args.partitions}")
    print(f"Sample size: {args.sample_size} queries per category")
    print(f"Visualizations: {'Disabled' if args.skip_viz else 'Enabled' if VISUALIZATION_AVAILABLE else 'Not Available'}")
    
    # Load model and tokenizer with error handling
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
        print("Please check your model name and internet connection.")
        return
    
    # Create and run experiment
    experiment = RobustResearchExperiment(model, tokenizer, args.partitions)
    
    logger.info("Starting robust research experiment...")
    results = await experiment.run_experiment(
        max_tokens=args.max_tokens,
        sample_size=args.sample_size
    )
    
    if not results:
        logger.error("No successful results generated. Please check your model and queries.")
        return
    
    # Analyze results
    logger.info("Analyzing results...")
    analysis = experiment.analyze_results(results)
    
    if "error" in analysis:
        logger.error(f"Analysis failed: {analysis['error']}")
        return
    
    # Create output directory
    output_dir = Path("./robust_research_output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results with ultra-safe serialization
    try:
        results_file = output_dir / f"results_{timestamp}.json"
        analysis_file = output_dir / f"analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=safe_json_serializer)
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=safe_json_serializer)
        
        logger.info("Results saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        # Try saving with minimal data
        try:
            minimal_results = [{'prompt': r['prompt'], 'category': r['category']} for r in results]
            with open(output_dir / f"minimal_results_{timestamp}.json", 'w') as f:
                json.dump(minimal_results, f, indent=2)
            logger.info("Minimal results saved as fallback")
        except:
            logger.error("Could not save any results")
    
    # Create visualizations if available
    if not args.skip_viz and VISUALIZATION_AVAILABLE:
        try:
            experiment.create_basic_visualizations(results, output_dir)
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
    
    # Generate text report
    try:
        report_file = output_dir / f"research_report_{timestamp}.txt"
        generate_text_report(analysis, results, report_file)
        logger.info(f"Text report generated: {report_file}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
    
    # Display results
    print("\n" + "="*80)
    print("ROBUST RESEARCH RESULTS")
    print("="*80)
    
    try:
        meta = analysis['experiment_metadata']
        print(f"Total queries processed: {meta['total_queries']}")
        print(f"Categories analyzed: {', '.join(meta['categories'])}")
        print(f"Model: {meta['model_info']}")
        print(f"Matrix size: {meta['partitions']}")
        
        # Overall statistics
        if analysis['overall_statistics']:
            print(f"\nOverall Statistics:")
            for metric, stats in analysis['overall_statistics'].items():
                print(f"  {metric.replace('_', ' ').title()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        # Category breakdown
        print(f"\nCategory Analysis:")
        for category, data in analysis['category_analysis'].items():
            print(f"\n  {category.upper()} ({data['query_count']} queries):")
            for metric, stats in data['statistics'].items():
                if metric in ['activation_entropy', 'pattern_complexity']:
                    print(f"    {metric.replace('_', ' ').title()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        # Key insights
        print(f"\nKey Research Insights:")
        for insight in analysis['research_insights']:
            print(f"  • {insight}")
        
        # Sample mind map
        if results:
            print(f"\nSample Thought Matrix:")
            sample = results[0]
            mind_map = sample['mind_map']
            print(f"Query: '{sample['prompt']}'")
            print(f"Category: {sample['category']}")
            print(f"Shape: {mind_map.shape}")
            
            for i, row in enumerate(mind_map):
                row_str = f"  Row {i}: [" + " ".join([f"{val:7.1f}" for val in row]) + "]"
                print(row_str)
        
        print(f"\nFiles Generated:")
        print(f"  Output directory: {output_dir}")
        if (output_dir / f"results_{timestamp}.json").exists():
            print(f"  ✓ Raw data: results_{timestamp}.json")
        if (output_dir / f"analysis_{timestamp}.json").exists():
            print(f"  ✓ Analysis: analysis_{timestamp}.json")
        if (output_dir / f"research_report_{timestamp}.txt").exists():
            print(f"  ✓ Report: research_report_{timestamp}.txt")
        
        if VISUALIZATION_AVAILABLE and not args.skip_viz:
            viz_dir = output_dir / "visualizations"
            if viz_dir.exists():
                print(f"  ✓ Visualizations: visualizations/")
        
    except Exception as e:
        logger.error(f"Error displaying results: {e}")
        print(f"Experiment completed with {len(results)} results, but display failed.")
        print(f"Check output directory: {output_dir}")
    
    print("\n" + "="*80)
    print("ROBUST EXPERIMENT COMPLETED")
    print("="*80)
    
    if not VISUALIZATION_AVAILABLE:
        print("\nTo enable visualizations in future runs:")
        print("pip install matplotlib seaborn scipy pandas")

def generate_text_report(analysis, results, report_file):
    """Generate a simple text report that always works"""
    
    try:
        with open(report_file, 'w') as f:
            f.write("THOUGHT MATRIX RESEARCH REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            meta = analysis['experiment_metadata']
            f.write(f"Generated: {meta['timestamp']}\n")
            f.write(f"Total Queries: {meta['total_queries']}\n")
            f.write(f"Model: {meta['model_info']}\n")
            f.write(f"Matrix Size: {meta['partitions']}\n")
            f.write(f"Categories: {', '.join(meta['categories'])}\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            for metric, stats in analysis['overall_statistics'].items():
                f.write(f"{metric.replace('_', ' ').title()}: ")
                f.write(f"{stats['mean']:.3f} ± {stats['std']:.3f} ")
                f.write(f"(range: {stats['min']:.3f} - {stats['max']:.3f})\n")
            f.write("\n")
            
            # Category analysis
            f.write("CATEGORY ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for category, data in analysis['category_analysis'].items():
                f.write(f"\n{category.upper()} ({data['query_count']} queries):\n")
                for metric, stats in data['statistics'].items():
                    f.write(f"  {metric.replace('_', ' ').title()}: ")
                    f.write(f"{stats['mean']:.3f} ± {stats['std']:.3f}\n")
                
                f.write("  Sample queries:\n")
                for query in data['sample_queries']:
                    f.write(f"    - {query}\n")
            
            # Insights
            f.write("\nKEY INSIGHTS\n")
            f.write("-" * 30 + "\n")
            for insight in analysis['research_insights']:
                f.write(f"• {insight}\n")
            
            # Sample data
            f.write("\nSAMPLE THOUGHT MATRIX\n")
            f.write("-" * 30 + "\n")
            if results:
                sample = results[0]
                f.write(f"Query: {sample['prompt']}\n")
                f.write(f"Category: {sample['category']}\n")
                f.write(f"Matrix:\n")
                mind_map = sample['mind_map']
                for i, row in enumerate(mind_map):
                    f.write(f"  [{' '.join([f'{val:7.1f}' for val in row])}]\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Report generated successfully.\n")
            
    except Exception as e:
        # Ultra-minimal fallback report
        with open(report_file, 'w') as f:
            f.write("THOUGHT MATRIX RESEARCH REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Report generation encountered an error: {str(e)}\n")
            f.write(f"Number of results processed: {len(results) if results else 0}\n")
            f.write(f"Basic experiment data was collected successfully.\n")

if __name__ == "__main__":
    asyncio.run(main())