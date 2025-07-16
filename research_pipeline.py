#!/usr/bin/env python3
"""
Research Pipeline for Thought Matrix Analysis
============================================

This script demonstrates how to run a complete research experiment
using the enhanced thought matrix system.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# Try to import enhanced components, fall back to basic if not available
try:
    from enhanced_thought_matrix import ResearchExperiment, QueryAnalyzer
    from research_analysis import analyze_experiment_results, generate_paper_sections
    from enhanced_config import Config, get_experiment_config
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced components not available: {e}")
    print("Using basic components instead...")
    ENHANCED_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchPipeline:
    """Complete research pipeline for thought matrix analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.results_dir = Path("./research_results")
        self.results_dir.mkdir(exist_ok=True)
        
    async def load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.config.MODEL_NAME}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            token=self.config.HF_TOKEN,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME, 
            token=self.config.HF_TOKEN
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Model and tokenizer loaded successfully")
    
    async def run_basic_experiment(self, max_tokens: int = 100) -> str:
        """Run basic cognitive assessment experiment"""
        logger.info("Starting basic cognitive assessment experiment")
        
        experiment = ResearchExperiment(
            self.model, 
            self.tokenizer, 
            self.config.N_PARTITIONS
        )
        
        results = await experiment.run_experiment(max_tokens)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"basic_experiment_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Basic experiment completed. Results saved to: {results_file}")
        return str(results_file)
    
    async def run_focused_experiment(self, categories: list, 
                                   sample_size: int = 15, 
                                   max_tokens: int = 150) -> str:
        """Run focused experiment on specific categories"""
        logger.info(f"Starting focused experiment on categories: {categories}")
        
        experiment = ResearchExperiment(
            self.model, 
            self.tokenizer, 
            self.config.N_PARTITIONS
        )
        
        # Generate test queries for specific categories
        all_queries = experiment.generate_test_queries()
        focused_queries = {cat: all_queries[cat][:sample_size] 
                          for cat in categories if cat in all_queries}
        
        # Run experiment with focused queries
        results = []
        category_results = {}
        
        for category, queries in focused_queries.items():
            category_results[category] = []
            logger.info(f"Processing {len(queries)} queries for category: {category}")
            
            for i, query in enumerate(queries):
                logger.info(f"  Query {i+1}/{len(queries)}: {query[:50]}...")
                
                result = await experiment.thought_matrix.infer_with_analysis(
                    query, self.tokenizer, max_tokens
                )
                
                results.append(result)
                category_results[category].append(result)
        
        # Analyze patterns
        pattern_analysis = experiment.query_analyzer.analyze_category_patterns(results)
        
        focused_results = {
            'experiment_type': 'focused',
            'categories_tested': categories,
            'sample_size_per_category': sample_size,
            'individual_results': results,
            'category_results': category_results,
            'pattern_analysis': pattern_analysis,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': self.config.MODEL_NAME,
                'partitions': self.config.N_PARTITIONS,
                'max_tokens': max_tokens
            }
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"focused_experiment_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(focused_results, f, indent=2, default=str)
        
        logger.info(f"Focused experiment completed. Results saved to: {results_file}")
        return str(results_file)
    
    async def run_comprehensive_analysis(self, results_file: str) -> dict:
        """Run comprehensive analysis on experiment results"""
        logger.info("Starting comprehensive analysis")
        
        # Create analysis output directory
        analysis_dir = self.results_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Run analysis
        analyzer, report = analyze_experiment_results(
            results_file, 
            str(analysis_dir)
        )
        
        # Generate paper sections
        paper_sections = generate_paper_sections(analyzer)
        
        # Save paper sections
        paper_file = analysis_dir / "paper_sections.json"
        with open(paper_file, 'w') as f:
            json.dump(paper_sections, f, indent=2)
        
        logger.info("Comprehensive analysis completed")
        return report
    
    def generate_research_summary(self, analysis_report: dict) -> str:
        """Generate a research summary for the paper"""
        
        summary = f"""
# Thought Matrix Research Summary

## Experiment Overview
- **Model**: {self.config.MODEL_NAME}
- **Total Samples**: {analysis_report['metadata']['total_samples']}
- **Categories**: {', '.join(analysis_report['metadata']['categories_analyzed'])}
- **Analysis Date**: {analysis_report['metadata']['analysis_date']}

## Key Findings
"""
        
        for finding in analysis_report['research_insights']['key_findings']:
            summary += f"- {finding}\n"
        
        summary += "\n## Statistical Results\n"
        
        # Add statistical test results
        for test_name, test_result in analysis_report['statistical_tests'].items():
            if isinstance(test_result, dict) and 'significant' in test_result:
                significance = "✓ Significant" if test_result['significant'] else "✗ Not significant"
                summary += f"- {test_name}: {significance} (p={test_result.get('p_value', 'N/A'):.4f})\n"
        
        summary += "\n## Research Implications\n"
        
        for implication in analysis_report['research_insights']['research_implications']:
            summary += f"- {implication}\n"
        
        return summary

async def main():
    """Main research pipeline execution"""
    parser = argparse.ArgumentParser(description="Run thought matrix research pipeline")
    parser.add_argument("--experiment", choices=["basic", "focused", "comprehensive"], 
                       default="basic", help="Type of experiment to run")
    parser.add_argument("--categories", nargs="+", 
                       default=["factual", "reasoning", "creative"],
                       help="Categories for focused experiment")
    parser.add_argument("--sample-size", type=int, default=10,
                       help="Sample size per category")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--analyze-existing", type=str,
                       help="Path to existing results file to analyze")
    
    args = parser.parse_args()
    
    # Initialize configuration
    try:
        config = Config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # Initialize pipeline
    pipeline = ResearchPipeline(config)
    
    try:
        if args.analyze_existing:
            # Analyze existing results
            logger.info(f"Analyzing existing results: {args.analyze_existing}")
            report = await pipeline.run_comprehensive_analysis(args.analyze_existing)
            
        else:
            # Load model first
            await pipeline.load_model()
            
            # Run experiment based on type
            if args.experiment == "basic":
                results_file = await pipeline.run_basic_experiment(args.max_tokens)
            elif args.experiment == "focused":
                results_file = await pipeline.run_focused_experiment(
                    args.categories, args.sample_size, args.max_tokens
                )
            elif args.experiment == "comprehensive":
                # Run both basic and focused experiments
                basic_file = await pipeline.run_basic_experiment(args.max_tokens)
                focused_file = await pipeline.run_focused_experiment(
                    ["reasoning", "creative", "mathematical"], 20, args.max_tokens
                )
                results_file = focused_file  # Analyze the focused one
            
            # Run analysis
            logger.info("Running comprehensive analysis...")
            report = await pipeline.run_comprehensive_analysis(results_file)
        
        # Generate and save summary
        summary = pipeline.generate_research_summary(report)
        summary_file = pipeline.results_dir / "research_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print("\n" + "="*60)
        print("RESEARCH PIPELINE COMPLETED")
        print("="*60)
        print(summary)
        print(f"\nDetailed results saved in: {pipeline.results_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise

def run_quick_demo():
    """Run a quick demonstration with a small model"""
    
    print("Starting Quick Demo...")
    
    # Use a small model for demo
    demo_config = Config()
    demo_config.MODEL_NAME = "gpt2"  # Small model for quick testing
    demo_config.N_PARTITIONS = 3
    demo_config.MAX_TOKENS = 50
    
    async def demo():
        pipeline = ResearchPipeline(demo_config)
        await pipeline.load_model()
        
        # Run small focused experiment
        results_file = await pipeline.run_focused_experiment(
            ["factual", "reasoning"], sample_size=3, max_tokens=30
        )
        
        # Quick analysis
        report = await pipeline.run_comprehensive_analysis(results_file)
        summary = pipeline.generate_research_summary(report)
        
        print("\n" + "="*50)
        print("QUICK DEMO RESULTS")
        print("="*50)
        print(summary)
    
    asyncio.run(demo())

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_quick_demo()
    else:
        asyncio.run(main())