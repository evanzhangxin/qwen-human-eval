# HumanEval Evaluation using Qwen3-0.6B

This repository provides instructions and scripts to serve the [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model using vLLM, and evaluate its performance on the [HumanEval](https://github.com/openai/human-eval) benchmark.

There are 3 steps in total for this task:

1. Serving: Serving the Qwen3-0.6B model using vLLM.
2. Inference: Generating code samples using the served Qwen3-0.6B model.
3. Evaluation: Evaluating the generated code samples using the HumanEval benchmark.

## Steps

### 1. Serving

#### 1.1 Build the Docker image

To build the Docker image for serving Qwen3-0.6B, run:

```bash
docker build -f ./dockerfiles/serving/Dockerfile.serve -t qwen3-server .
```

#### 1.2 Run model serving (host the image)

Use **MODEL_PATH** pointing to your local model directory, then run the container mounting that directory with -v:

```bash
# set local model path (replace with your actual path)
export MODEL_PATH="/path/to/model_cache"
```

Serve Qwen3 0.6B using vLLM on GPU
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
```
1. Replace MODEL_PATH with your local path contains Qwen3-0.6B model files.
2. Adjust ports, environment variables, and volume mounts as needed for your deployment.
```
#### 1.3 Test the service with curl

Quick test using vLLM completions endpoint, you can replace content in messages and other parameters like temperature etc.

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


### 2.Inference

#### 2.1 Install dependencies
Currently still using humaneval dataset loading function to load and process samples, so need to install humaneval package first.
```bash
pip install human-eval
```
Note: Check [human-eval](https://github.com/openai/human-eval) for more details.


#### 2.2 Run Inference
All the dataset loading, inference method, and post-processing functions are packed in inference.py right now, you can run it directly.
```bash
python scripts/inference.py
```
Note:
1. There are few assumptions here, the localhost:8000 is the default port for vLLM serving, and there will be a default prompt for better instruct following.
```
'You are an intelligent programming assistant to produce Python algorithmic solutions.\nCan you complete the following Python function?\n```python\n{prompt}\n```'
```
2. the post-processing function is to extract the code from the response. The original code is from [OpenCompass](https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/humaneval.py), it might normalize some formatting issues caused by model which could lead to a higher score.

#### 2.3 Use Evaluation container for inference
You can also use the evaluation container to run inference, just make sure to mount a directory to save the results file. Details about the container can be found in section 3.
```bash
python scripts/inference.py
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

Then, inside the container, run the evaluation script, it will score the results file and dump the overall result along with scores for each sample in a seperated file: 

```bash 
evaluate_functional_correctness /app/results/results.jsonl
```
The evaluation results will be saved in the same directory as the results file.

Note: If you want to use different sample files or update other parameters, you can run the command to get help:

```bash
evaluate_functional_correctness --help
```

#### 3.3 Evaluation Results
1. Evaluation Results for results/samples_full_10.jsonl 
```json
{
    "pass@1": np.float64(0.32743902439024386),
    "pass@5": np.float64(0.5432152535811072),
    "pass@10": np.float64(0.6280487804878049)
}
```

2. Evaluation Results for results/samples_full_50.jsonl
```json
{
    'pass@1': np.float64(0.33341463414634154), 
    'pass@10': np.float64(0.629393448578781), 
    'pass@20': np.float64(0.6948370262211264), 
    'pass@50': np.float64(0.7682926829268293)
}
```
## Proposal for Benchmark Improvement

As shown below each benchmark targets a distinct aspect of "coding ability": from basic function generation (HumanEval/MBPP) to algorithmic problem-solving (Codeforces) and real-world software development (LiveCodeBench/SWE-bench).

So for metric improvement itself, it would be necessary that consider about the code quality in addtion to Pass@K, since the answer might be correct but the solution is not very practical or userful.

| **Dimension** | **HumanEval** | **MBPP** | **Codeforces-style** | **LiveCodeBench** | **SWE-bench** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task Origin** | Handcrafted | Crowdsourced  | Competitive programming problems  | Real-world dev tasks (bug fixes, feature additions). | Real GitHub issues from large OSS projects. |
| **Task Complexity** | Low to medium: Short Python functions (~10 lines) with clear specs. | Low: Very basic Python functions (e.g., "reverse a list") with step-by-step instructions. | High: Algorithmic puzzles, math/logic challenges, time/space constraints. | Medium to high: Multi-file projects, dependencies, real workflows. | Very high: Large codebases, ambiguous issues, integration constraints. |
| **Key Metrics** | pass@k (functional correctness via test cases). | pass@k (similar to HumanEval, with simpler tests). | Accuracy (solving contest problems), efficiency (runtime/memory). | Functional correctness + code quality (readability, style, efficiency). | Issue resolution (does code fix the problem?), CI/test suite passing. |
| **Languages** | Exclusively Python. | Exclusively Python. | Multi-lingual (C++, Python, Java, etc.). | Multi-lingual (10+ languages: Python, JS, Java, etc.). | Multi-lingual (matches OSS project languages). |
| **Test Cases** | Small, handwritten test suites. | Crowdsourced test cases (simpler, more verbose). | Rigorous, edge-case-heavy test suites (critical for contest fairness). | Project-specific tests + static analysis + human evaluation. | Original project test suites + real-world usage constraints. |
| **Realism** | Low: Isolated functions, no context beyond docstrings. | Low: Trivial tasks, minimal real-world context. | Medium: Algorithmic challenges reflect problem-solving but not software engineering. | High: Mimics daily dev work (dependencies, code style, collaboration). | Very high: Directly mirrors real-world software maintenance. |


`pass@k` provides a simple baseline for measuring functional correctness in code generation, but still there are few places can be improved IMO:

1. measure code quality in addtion to code pass rate
2. measure more details e.g. error types (syntax or logical flaws and so on) to figure out which tasks LLM doing not well, and to guide people to focus on it
3. add more test cases for better coverage



Other Proposals(to Benchmark)：

**Realism**: better to align with practical development

**Comprehensive**: this is quite big topic, the differnet coding tasks, context complexity, cross-domain questions, different engineering practices like how to design, debug, implement, test etc. All these can be created as evaluation tasks.

**Dynamism**: <font style="color:rgba(0, 0, 0, 0.85);">adaptation to technological advancements and also can prevent data leakage in some degree</font>


## Possible Evaluation Performance Improvement
Have tried the batching method on vLLM, enabling parallel inference when parameters such as max_token can afford. If there are multiple GPUs or a sufficiently powerful GPU with large RAM, multiple instances can be served on different or same GPU card. This allows the dataset to be split using data parallelism and enables multi-process execution.

Additionally, there hasn't been time to try using quantization for inference, but it's likely that there will be a loss in performance.

Qwen3 supports turning on/off the thinking mode, it would be a good idea to compare the differnece to get more insights.</font>




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

