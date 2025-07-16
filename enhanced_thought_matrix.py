import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import json
import re
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import uuid

@dataclass
class ResearchMetrics:
    """Container for research-specific metrics"""
    activation_entropy: float
    region_specialization: float
    cross_region_connectivity: float
    pattern_complexity: float
    category_signature: np.ndarray
    temporal_dynamics: Dict[str, float]
    functional_distribution: Dict[str, float]

class QueryAnalyzer:
    """Analyzes and categorizes queries for research purposes"""
    
    def __init__(self):
        self.categories = {
            'factual': [
                r'what is', r'define', r'when did', r'who is', r'where is',
                r'how many', r'what are', r'which', r'name the'
            ],
            'reasoning': [
                r'why', r'how does', r'explain', r'what causes', r'what happens if',
                r'analyze', r'compare', r'relationship between', r'implications of'
            ],
            'creative': [
                r'write a', r'imagine', r'create', r'story about', r'poem about',
                r'design', r'invent', r'pretend', r'roleplay'
            ],
            'mathematical': [
                r'calculate', r'solve', r'what is.*\+', r'what is.*\-', r'what is.*\*',
                r'find the', r'prove', r'derive', r'integral of', r'derivative of'
            ],
            'ethical': [
                r'should', r'is it right', r'moral', r'ethical', r'ought to',
                r'implications', r'consequences', r'fair', r'just'
            ],
            'conversational': [
                r'hello', r'hi', r'how are you', r'thanks', r'please',
                r'can you help', r'i need', r'would you'
            ]
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) 
                      for pattern in patterns]
            for category, patterns in self.categories.items()
        }
    
    def categorize_query(self, prompt: str) -> str:
        """Categorize a query based on patterns"""
        prompt_lower = prompt.lower()
        
        # Score each category
        category_scores = {}
        for category, patterns in self.compiled_patterns.items():
            score = sum(1 for pattern in patterns if pattern.search(prompt_lower))
            if score > 0:
                category_scores[category] = score
        
        if not category_scores:
            return 'other'
        
        # Return category with highest score
        return max(category_scores, key=category_scores.get)
    
    def analyze_category_patterns(self, results: List[Dict]) -> Dict:
        """Analyze activation patterns by query category"""
        category_patterns = defaultdict(list)
        
        for result in results:
            category = self.categorize_query(result['prompt'])
            category_patterns[category].append(result['mind_map'])
        
        analysis = {}
        for category, patterns in category_patterns.items():
            if len(patterns) > 1:
                patterns_array = np.array(patterns)
                analysis[category] = {
                    'mean_pattern': np.mean(patterns_array, axis=0),
                    'std_pattern': np.std(patterns_array, axis=0),
                    'pattern_consistency': self._calculate_consistency(patterns_array),
                    'sample_size': len(patterns),
                    'unique_regions': self._identify_unique_regions(patterns_array)
                }
        
        return analysis
    
    def _calculate_consistency(self, patterns: np.ndarray) -> float:
        """Calculate pattern consistency using pairwise cosine similarity"""
        if len(patterns) < 2:
            return 1.0
        
        # Flatten patterns for similarity calculation
        flat_patterns = patterns.reshape(len(patterns), -1)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(flat_patterns)):
            for j in range(i+1, len(flat_patterns)):
                sim = cosine_similarity([flat_patterns[i]], [flat_patterns[j]])[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _identify_unique_regions(self, patterns: np.ndarray) -> List[Tuple[int, int]]:
        """Identify regions that are uniquely active for this category"""
        mean_pattern = np.mean(patterns, axis=0)
        threshold = np.percentile(mean_pattern, 75)  # Top 25% of activations
        
        unique_regions = []
        for i in range(mean_pattern.shape[0]):
            for j in range(mean_pattern.shape[1]):
                if mean_pattern[i, j] > threshold:
                    unique_regions.append((i, j))
        
        return unique_regions

class EnhancedThoughtMatrix:
    """Enhanced thought matrix with functional partitioning and detailed analysis"""
    
    def __init__(self, model, n_partitions=4):
        self.model = model
        self.n_partitions = n_partitions
        self.device = next(model.parameters()).device
        
        # Initialize tracking structures
        self.mind_map = np.zeros((n_partitions, n_partitions))
        self.token_maps = [[[] for _ in range(n_partitions)] for _ in range(n_partitions)]
        self.activation_variance = np.zeros((n_partitions, n_partitions))
        self.temporal_patterns = []
        
        # Set up functional partitioning
        self.functional_regions = self._setup_functional_partitions()
        self.partition_assignments = self._create_partition_assignments()
        
        # Hooks management
        self.activation_hooks = []
        self.query_analyzer = QueryAnalyzer()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_functional_partitions(self) -> Dict[str, List[nn.Module]]:
        """Group layers by function rather than arbitrary splitting"""
        attention_layers = []
        mlp_layers = []
        embedding_layers = []
        norm_layers = []
        other_layers = []
        
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ['attn', 'attention', 'self_attn']):
                attention_layers.append((name, module))
            elif any(x in name.lower() for x in ['mlp', 'ffn', 'feed_forward']):
                mlp_layers.append((name, module))
            elif any(x in name.lower() for x in ['embed', 'wte', 'wpe']):
                embedding_layers.append((name, module))
            elif any(x in name.lower() for x in ['norm', 'ln', 'layer_norm']):
                norm_layers.append((name, module))
            elif isinstance(module, nn.Linear):
                other_layers.append((name, module))
        
        self.logger.info(f"Functional partitions: "
                        f"Attention: {len(attention_layers)}, "
                        f"MLP: {len(mlp_layers)}, "
                        f"Embedding: {len(embedding_layers)}, "
                        f"Norm: {len(norm_layers)}, "
                        f"Other: {len(other_layers)}")
        
        return {
            'attention': attention_layers,
            'mlp': mlp_layers,
            'embedding': embedding_layers,
            'norm': norm_layers,
            'other': other_layers
        }
    
    def _create_partition_assignments(self) -> np.ndarray:
        """Assign functional regions to grid partitions"""
        total_partitions = self.n_partitions ** 2
        assignments = np.empty((self.n_partitions, self.n_partitions), dtype=object)
        
        # Collect all modules with their functional labels
        all_modules = []
        for func_type, modules in self.functional_regions.items():
            for name, module in modules:
                all_modules.append({
                    'name': name,
                    'module': module,
                    'function': func_type
                })
        
        # Distribute modules across partitions
        if len(all_modules) < total_partitions:
            # If we have fewer modules than partitions, replicate
            all_modules = all_modules * ((total_partitions // len(all_modules)) + 1)
        
        # Assign to grid positions
        module_chunks = np.array_split(all_modules[:total_partitions], total_partitions)
        
        idx = 0
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                assignments[i, j] = module_chunks[idx]
                idx += 1
        
        return assignments
    
    def _attach_enhanced_hooks(self):
        """Attach hooks with enhanced tracking capabilities"""
        self.mind_map.fill(0)
        self.activation_variance.fill(0)
        self.token_maps = [[[] for _ in range(self.n_partitions)] 
                          for _ in range(self.n_partitions)]
        self.temporal_patterns = []
        
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                modules = self.partition_assignments[i, j]
                
                for module_info in modules:
                    module = module_info['module']
                    
                    def make_hook(i, j, func_type):
                        def hook(module, input, output):
                            try:
                                # Handle different output types
                                if isinstance(output, tuple):
                                    actual_output = output[0]
                                else:
                                    actual_output = output
                                
                                if actual_output is None:
                                    return
                                
                                # Track per-token activations
                                if len(actual_output.shape) >= 2:
                                    # [batch, seq_len, hidden] or [batch, seq_len]
                                    token_activations = actual_output.abs().mean(dim=-1)
                                    
                                    # Aggregate metrics
                                    mean_activation = token_activations.mean().item()
                                    var_activation = token_activations.var().item()
                                    
                                    self.mind_map[i, j] += mean_activation
                                    self.activation_variance[i, j] += var_activation
                                    
                                    # Store temporal pattern
                                    self.token_maps[i][j].append({
                                        'activations': token_activations.cpu().numpy(),
                                        'function': func_type,
                                        'timestamp': len(self.temporal_patterns)
                                    })
                                
                            except Exception as e:
                                self.logger.warning(f"Hook error at [{i},{j}]: {e}")
                        
                        return hook
                    
                    hook = make_hook(i, j, module_info['function'])
                    handle = module.register_forward_hook(hook)
                    self.activation_hooks.append(handle)
    
    def _detach_hooks(self):
        """Clean up all hooks"""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks.clear()
    
    def calculate_research_metrics(self, mind_map: np.ndarray, 
                                 query_category: str) -> ResearchMetrics:
        """Calculate comprehensive research metrics"""
        
        # 1. Activation Entropy
        flat = mind_map.flatten()
        flat = flat / (flat.sum() + 1e-10)  # Normalize to probability distribution
        activation_entropy = entropy(flat + 1e-10, base=2)
        
        # 2. Region Specialization (how concentrated activations are)
        region_specialization = np.var(mind_map) / (np.mean(mind_map) + 1e-10)
        
        # 3. Cross-region Connectivity
        cross_region_connectivity = self._calculate_connectivity(mind_map)
        
        # 4. Pattern Complexity
        pattern_complexity = self._calculate_complexity(mind_map)
        
        # 5. Category Signature
        category_signature = self._extract_category_signature(mind_map, query_category)
        
        # 6. Temporal Dynamics
        temporal_dynamics = self._analyze_temporal_dynamics()
        
        # 7. Functional Distribution
        functional_distribution = self._analyze_functional_distribution()
        
        return ResearchMetrics(
            activation_entropy=activation_entropy,
            region_specialization=region_specialization,
            cross_region_connectivity=cross_region_connectivity,
            pattern_complexity=pattern_complexity,
            category_signature=category_signature,
            temporal_dynamics=temporal_dynamics,
            functional_distribution=functional_distribution
        )
    
    def _calculate_connectivity(self, matrix: np.ndarray) -> float:
        """Calculate cross-region connectivity using correlation"""
        if matrix.size == 0:
            return 0.0
        
        # Calculate correlation between adjacent regions
        correlations = []
        for i in range(matrix.shape[0] - 1):
            for j in range(matrix.shape[1] - 1):
                # Horizontal connectivity
                if j + 1 < matrix.shape[1]:
                    corr = np.corrcoef(matrix[i, :j+2])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                
                # Vertical connectivity
                if i + 1 < matrix.shape[0]:
                    corr = np.corrcoef(matrix[:i+2, j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_complexity(self, matrix: np.ndarray) -> float:
        """Calculate pattern complexity using local variance"""
        if matrix.size <= 1:
            return 0.0
        
        # Use gradient magnitude as complexity measure
        grad_x = np.gradient(matrix, axis=0)
        grad_y = np.gradient(matrix, axis=1)
        complexity = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        return complexity
    
    def _extract_category_signature(self, matrix: np.ndarray, 
                                  category: str) -> np.ndarray:
        """Extract category-specific signature from activation pattern"""
        # Normalize matrix to create signature
        if matrix.sum() == 0:
            return matrix.copy()
        
        signature = matrix / matrix.sum()
        
        # Apply category-specific weighting (this could be learned)
        category_weights = {
            'factual': np.array([[1.2, 1.0, 0.8, 0.6],
                                [1.0, 1.1, 0.9, 0.7],
                                [0.8, 0.9, 1.0, 0.8],
                                [0.6, 0.7, 0.8, 1.0]]),
            'reasoning': np.array([[0.8, 1.0, 1.2, 1.1],
                                 [1.0, 1.2, 1.3, 1.0],
                                 [1.2, 1.3, 1.1, 0.9],
                                 [1.1, 1.0, 0.9, 0.8]]),
            'creative': np.array([[1.0, 1.1, 1.2, 1.3],
                                [1.1, 1.2, 1.3, 1.2],
                                [1.2, 1.3, 1.2, 1.1],
                                [1.3, 1.2, 1.1, 1.0]]),
            'mathematical': np.array([[1.3, 1.2, 1.0, 0.8],
                                    [1.2, 1.3, 1.1, 0.9],
                                    [1.0, 1.1, 1.2, 1.0],
                                    [0.8, 0.9, 1.0, 1.1]])
        }
        
        # Resize weight matrix to match mind_map dimensions
        if category in category_weights:
            weights = category_weights[category]
            if weights.shape != matrix.shape:
                from scipy.ndimage import zoom
                scale_factors = [matrix.shape[i] / weights.shape[i] 
                               for i in range(len(matrix.shape))]
                weights = zoom(weights, scale_factors)
            
            signature = signature * weights
        
        return signature
    
    def _analyze_temporal_dynamics(self) -> Dict[str, float]:
        """Analyze temporal patterns in activations"""
        if not self.token_maps:
            return {'stability': 0.0, 'progression': 0.0}
        
        # Calculate stability across time steps
        temporal_variances = []
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                if self.token_maps[i][j]:
                    activations = [tm['activations'] for tm in self.token_maps[i][j]]
                    if len(activations) > 1:
                        # Calculate variance across time steps
                        time_var = np.var([np.mean(act) for act in activations])
                        temporal_variances.append(time_var)
        
        stability = 1.0 / (1.0 + np.mean(temporal_variances)) if temporal_variances else 0.0
        
        # Calculate progression (how activation changes over time)
        progression_scores = []
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                if len(self.token_maps[i][j]) > 1:
                    activations = [np.mean(tm['activations']) for tm in self.token_maps[i][j]]
                    if len(activations) > 1:
                        # Linear trend in activations
                        x = np.arange(len(activations))
                        progression = np.corrcoef(x, activations)[0, 1]
                        if not np.isnan(progression):
                            progression_scores.append(abs(progression))
        
        progression = np.mean(progression_scores) if progression_scores else 0.0
        
        return {
            'stability': stability,
            'progression': progression
        }
    
    def _analyze_functional_distribution(self) -> Dict[str, float]:
        """Analyze how activations are distributed across functional regions"""
        function_activations = defaultdict(list)
        
        for i in range(self.n_partitions):
            for j in range(self.n_partitions):
                if self.token_maps[i][j]:
                    for tm in self.token_maps[i][j]:
                        function_activations[tm['function']].append(np.mean(tm['activations']))
        
        # Calculate distribution metrics
        distribution = {}
        total_activation = sum(sum(acts) for acts in function_activations.values())
        
        for func_type, activations in function_activations.items():
            if activations:
                distribution[func_type] = sum(activations) / total_activation
        
        return distribution
    
    async def infer_with_analysis(self, prompt: str, tokenizer, max_tokens: int = 100) -> Dict[str, Any]:
        """Run inference with comprehensive analysis"""
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Set up hooks
            self._attach_enhanced_hooks()
            
            # Run inference
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
                self.mind_map = self.mind_map / np.max(self.mind_map)
            
            # Decode output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Analyze query category
            query_category = self.query_analyzer.categorize_query(prompt)
            
            # Calculate research metrics
            metrics = self.calculate_research_metrics(self.mind_map, query_category)
            
            return {
                'prompt': prompt,
                'output': output_text,
                'mind_map': self.mind_map.copy(),
                'query_category': query_category,
                'metrics': metrics,
                'activation_variance': self.activation_variance.copy(),
                'functional_regions': list(self.functional_regions.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            self._detach_hooks()

class ResearchExperiment:
    """Orchestrates research experiments with systematic data collection"""
    
    def __init__(self, model, tokenizer, n_partitions=4):
        self.thought_matrix = EnhancedThoughtMatrix(model, n_partitions)
        self.tokenizer = tokenizer
        self.results = []
        self.query_analyzer = QueryAnalyzer()
        
    def generate_test_queries(self) -> Dict[str, List[str]]:
        """Generate balanced test queries for each category"""
        test_queries = {
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
                "Who wrote Romeo and Juliet?"
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
                "How does democracy differ from autocracy?"
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
                "Design a perfect city."
            ],
            'mathematical': [
                "Calculate 17 × 23.",
                "What is 25% of 200?",
                "Solve for x: 2x + 5 = 15.",
                "Find the area of a circle with radius 7.",
                "What is the square root of 144?",
                "Calculate the compound interest on $1000 at 5% for 3 years.",
                "Find the derivative of x².",
                "What is 15! (15 factorial)?",
                "Solve the equation: x² - 5x + 6 = 0.",
                "Calculate the probability of rolling a 6 on a die."
            ],
            'ethical': [
                "Should artificial intelligence have rights?",
                "Is it ethical to test on animals?",
                "What are the moral implications of genetic engineering?",
                "Should wealthy nations help poor ones?",
                "Is capital punishment justified?",
                "What are the ethics of privacy vs security?",
                "Should we colonize Mars?",
                "Is it fair to have different wages for different jobs?",
                "What are the ethics of autonomous weapons?",
                "Should we extend human lifespan indefinitely?"
            ],
            'conversational': [
                "Hello, how are you today?",
                "Can you help me with something?",
                "Thank you for your assistance.",
                "Please explain this to me.",
                "I'm feeling confused about this topic.",
                "Would you mind clarifying that?",
                "I appreciate your help.",
                "Could you give me some advice?",
                "What do you think about this?",
                "I need your opinion on something."
            ]
        }
        
        return test_queries
    
    async def run_experiment(self, max_tokens: int = 50) -> Dict[str, Any]:
        """Run comprehensive experiment across all query categories"""
        test_queries = self.generate_test_queries()
        
        self.results = []
        category_results = {}
        
        for category, queries in test_queries.items():
            category_results[category] = []
            
            for query in queries:
                result = await self.thought_matrix.infer_with_analysis(
                    query, self.tokenizer, max_tokens
                )
                
                self.results.append(result)
                category_results[category].append(result)
        
        # Analyze patterns across categories
        pattern_analysis = self.query_analyzer.analyze_category_patterns(self.results)
        
        # Calculate cross-category metrics
        cross_category_metrics = self._calculate_cross_category_metrics(category_results)
        
        return {
            'individual_results': self.results,
            'category_results': category_results,
            'pattern_analysis': pattern_analysis,
            'cross_category_metrics': cross_category_metrics,
            'experiment_metadata': {
                'total_queries': len(self.results),
                'categories': list(test_queries.keys()),
                'n_partitions': self.thought_matrix.n_partitions,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _calculate_cross_category_metrics(self, category_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate metrics comparing different categories"""
        cross_metrics = {}
        
        # Calculate category distinctiveness
        category_signatures = {}
        for category, results in category_results.items():
            if results:
                mind_maps = [r['mind_map'] for r in results]
                category_signatures[category] = np.mean(mind_maps, axis=0)
        
        # Calculate pairwise similarities between categories
        similarities = {}
        categories = list(category_signatures.keys())
        
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories[i+1:], i+1):
                sig1 = category_signatures[cat1].flatten()
                sig2 = category_signatures[cat2].flatten()
                
                similarity = cosine_similarity([sig1], [sig2])[0][0]
                similarities[f"{cat1}_vs_{cat2}"] = similarity
        
        cross_metrics['category_similarities'] = similarities
        cross_metrics['category_signatures'] = {
            cat: sig.tolist() for cat, sig in category_signatures.items()
        }
        
        return cross_metrics
    
    def export_results(self, filename: str = None) -> str:
        """Export results to JSON file"""
        if filename is None:
            filename = f"thought_matrix_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        exportable_results = []
        for result in self.results:
            exportable_result = result.copy()
            exportable_result['mind_map'] = result['mind_map'].tolist()
            exportable_result['activation_variance'] = result['activation_variance'].tolist()
            
            # Convert metrics to dict
            metrics = result['metrics']
            exportable_result['metrics'] = {
                'activation_entropy': metrics.activation_entropy,
                'region_specialization': metrics.region_specialization,
                'cross_region_connectivity': metrics.cross_region_connectivity,
                'pattern_complexity': metrics.pattern_complexity,
                'category_signature': metrics.category_signature.tolist(),
                'temporal_dynamics': metrics.temporal_dynamics,
                'functional_distribution': metrics.functional_distribution
            }
            
            exportable_results.append(exportable_result)
        
        with open(filename, 'w') as f:
            json.dump(exportable_results, f, indent=2)
        
        return filename