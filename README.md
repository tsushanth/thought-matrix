# thought-matrix
Core Inference Libraries
PyTorch/TensorFlow (Base)

torch, transformers (Hugging Face)

For low-level model partitioning and execution

Specialized Inference

vllm (High-throughput serving)

llama.cpp (GGUF quantized models)

text-generation-inference (Hugging Face's production server)

Key Components
Model Partitioning

Split model into NxN logical blocks

Use PyTorch's module.children() for layer access

Alternative: Split by parameter groups

Activation Tracking

Forward hooks to record which partitions activate

Heatmap generation via mind_map accumulation

Evaluation System

Secondary LLM evaluates thought patterns

Customizable evaluation criteria

Correlates activation patterns with qualities

How to Run locally:

docker build  -t thought-evaluator-local .

docker run -d \                           
  -p 8080:8080 \               
  -e PORT=8080 -e HF_TOKEN=<your huggingface token> \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  thought-evaluator-local
