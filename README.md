# HumanEval Evaluation using Qwen3-0.6B

This repository provides instructions and scripts to serve the [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model using vLLM, and evaluate its performance on the [HumanEval](https://github.com/openai/human-eval) benchmark.

## Steps

### 1. Build and run Docker image for model serving

#### 1.1 Build the Docker image

To build the Docker image for serving Qwen3-0.6B, run:

```bash
docker build -f ./dockerfiles/serving/Dockerfile.serve -t qwen3-server .
```

#### 1.2 Run model serving (host the image)

- Use **MODEL_PATH** pointing to your local model directory, then run the container mounting that directory with -v:

```bash
# set local model path (replace with your actual path)
export MODEL_PATH="/path/to/model_cache"
```

- Serve Qwen3 0.6B using vLLM on GPU
```bash
docker run --gpus all -d --name qwen3-server \
    -p 8000:8000 \
    -v "${MODEL_PATH}":/model \
    qwen3-server /model \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.6 \
    --max-seq-len-to-capture 2048 \
    --max-model-len 8192 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --trust-remote-code
```

Notes:
- Replace MODEL_PATH with your local path to cache or store model files.
- Adjust ports, environment variables, and volume mounts as needed for your deployment.

### 1.3 Test the service with curl

Quick test using an OpenAI-compatible completions endpoint. Replace the prompt, model name, or endpoint URL as needed.

Example:
```bash
curl -s -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
         "messages": [
            {"role": "user", "content": "Give me a short introduction to large language models."}
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 1024,
        "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```


## 2.Inference


1. **Build and Run Docker Container**
    ```bash
    docker build -t qwen3-server .
    docker run --gpus all -p 8000:8000 qwen3-server
    ```

2. **Run Inference**
    ```bash
    python infer_humaneval.py --model-endpoint http://localhost:8000
    ```

3. **Evaluate Results**
    ```bash
    python evaluate_humaneval.py --results results.json
    ```

## 3. Evaluation

#### 3.1 Build and Run Evaluation Docker Image
To build the Docker image for evaluation, run:

```bash
docker build -f ./dockerfiles/evaluation/Dockerfile.eval -t humaneval-eval .
```
Then run the container, mount the results directory to access the results file:

```bash
docker run -d -it --name humaneval-eval \
    -v /path/to/results:/app/results \
    humaneval-eval \
    /bin/bash
``` 
Make sure to replace `/path/to/results` with the actual directory path to your results file.

#### 3.2 Run Evaluation Script

First, make sure the results file is in the mounted directory. You can check the mounted directory by running:

```bash
docker exec -it humaneval-eval bash
ls /app/results
```

Then, inside the container, run the evaluation script: 

```bash 
evaluate_functional_correctness /app/results/results.jsonl
```
The evaluation results will be saved in the same directory as the results file.

Note: If you want to use different sample files or update other parameters, you can run the command to get help:

```bash
evaluate_functional_correctness --help
```


## Requirements

- Docker
- Python 3.8+
- GPU (optional, recommended for faster inference)
- vLLM or SgLang
- HumanEval dataset

## References

- [Qwen/Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [HumanEval Benchmark](https://github.com/openai/human-eval)
- [vLLM](https://github.com/vllm-project/vllm) & [SgLang](https://github.com/sglang/sglang)
- [Qwen vLLM Deployment Guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)

