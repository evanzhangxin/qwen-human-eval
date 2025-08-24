import json
import re
import requests
from tqdm import tqdm

from human_eval.data import write_jsonl, read_problems

# This function is copied from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/humaneval.py, which is for extracting runnable code from chat response.
def humaneval_internal_v1_postprocess(text: str) -> str:
    """This is an advanced version of previous postprocess to handle more
    situations, better to use this one."""
    try:
        # for chatGLM related text
        eval_text = eval(text)
    except Exception:
        pass
    else:
        if isinstance(eval_text, str):
            text = eval_text
    text = text.lstrip('\n')
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split('```')[1]  # fall back to default strategy
        else:
            text = blocks[0]  # fetch the first code block
            if not text.startswith('\n'):  # in case starting with ```python
                text = text[max(text.find('\n') + 1, 0) :]
    if text.strip().startswith('from') or text.strip().startswith('import'):
        def_idx = text.find('def')
        if def_idx != -1:
            text = text[max(text.find('\n', def_idx) + 1, 0) :]
    # remove empty lines
    text = '\n'.join([line for line in text.split('\n') if line != ''])
    text = text.lstrip('\n')
    if text.strip().startswith('def'):
        text = '\n'.join(text.split('\n')[1:])
    # deal with the indentation error
    if text.startswith(' '):
        text = '    ' + text.lstrip()
    else:
        text = '\n'.join(['    ' + line for line in text.split('\n')])
    text = text.split('\n')

    # If number of leading space reduces, we assume that the code block ends.
    min_leading_space = None
    end_index = None
    for index, line in enumerate(text):
        if line.strip() == '' or line.strip()[0] in ["'", '"', '#']:
            continue
        current_leading_space = len(line.rstrip()) - len(line.strip())
        if min_leading_space is None:
            min_leading_space = current_leading_space
        elif current_leading_space < min_leading_space:
            end_index = index
            break
    if end_index is not None:
        text = '\n'.join(text[:end_index])
    else:
        text = '\n'.join(text)
    return text


def chat_with_qwen(prompt, enable_thinking=False, temperature=1.0, max_tokens=2048, top_p=0.95, top_k=20):
    """
    Sends a prompt to the Qwen language model service and returns the processed response.
    
    Args:
        prompt (str): The Python code prompt/function to be completed
        enable_thinking (bool): Whether to enable chain-of-thought reasoning in responses
        temperature (float): Sampling temperature (0.0-2.0), higher values increase randomness
        max_tokens (int): Maximum number of tokens to generate in response
        top_p (float): Nucleus sampling probability threshold (0.0-1.0)
        top_k (int): Number of highest probability tokens to consider for sampling
    
    Returns:
        str: Processed code response from Qwen service, or None if request fails
    """
    # API endpoint of the local Qwen vLLM service (configured in serve_qwen.py)
    url = "http://localhost:8000/v1/chat/completions"

    # Request headers specifying JSON content type
    headers = {
        "Content-Type": "application/json"
    }

    # Wrap prompt in a programming assistant system message template
    prompt = 'You are an intelligent programming assistant to produce Python algorithmic solutions.\nCan you complete the following Python function?\n```python\n{prompt}\n```'.format(prompt=prompt)
    
    # Request payload following vLLM chat completions API format
    payload = {
        "messages": [
            {"role": "user", "content": prompt}  # User message with formatted prompt
        ],
        "temperature": temperature,  # Controls randomness of generation
        "max_tokens": max_tokens,    # Maximum response length
        "top_p": top_p,              # Nucleus sampling parameter
        "top_k": top_k,              # Top-K sampling parameter
        "chat_template_kwargs": {"enable_thinking": enable_thinking}  # Qwen-specific thinking mode
    }

    try:
        # Send POST request to Qwen service with JSON payload
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise HTTP errors (4xx/5xx responses)

        # Parse JSON response and extract generated content
        result = response.json()
        response = result["choices"][0]["message"]["content"]
        
        # Post-process response to extract valid Python code (custom implementation)
        response_impl = humaneval_internal_v1_postprocess(response)
        return response_impl

    except requests.exceptions.RequestException as e:
        # Handle request errors (connection issues, timeouts, API errors)
        print(f"Error communicating with Qwen service: {e}")
        return None

def generate_samples(problems, num_samples_per_task, output_file):
    """
    Generates code samples for given problems using Qwen model and saves results to JSONL file.
    
    Args:
        problems (dict): Dictionary of problem objects with task IDs as keys
        num_samples_per_task (int): Number of code samples to generate per problem
        output_file (str): Path to output JSONL file where samples will be saved
    
    Returns:
        None: Results are saved directly to output_file
    """
    samples = []
    # Calculate total samples and create progress bar
    total_samples = len(problems) * num_samples_per_task
    with tqdm(total=total_samples, desc="Sample Generation Progress", unit="sample") as pbar:
        for task_id in problems:
            # Extract prompt for current problem
            problem_prompt = problems[task_id]["prompt"]
            
            # Generate specified number of samples for current problem
            for _ in range(num_samples_per_task):
                # Call Qwen model to generate code completion
                completion = chat_with_qwen(problem_prompt)
                
                # Store sample with task ID and generated completion
                samples.append(dict(task_id=task_id, completion=completion))
                
                # Update progress bar after each sample
                pbar.update(1)
    
    # Write all generated samples to JSONL file
    write_jsonl(output_file, samples)


def main():
    """Main function to execute sample generation workflow"""
    # Configure generation parameters
    NUM_SAMPLES_PER_TASK = 10  # Number of samples to generate per problem
    OUTPUT_FILE = "samples_generated.jsonl"  # Output file path

    # Load problems from HumanEval dataset
    print("Loading problem dataset...")
    problems = read_problems()
    print(f"Loaded {len(problems)} problems successfully")

    # Generate and save samples
    print(f"Starting sample generation ({{NUM_SAMPLES_PER_TASK}} samples per problem)...")
    generate_samples(
        problems=problems,
        num_samples_per_task=NUM_SAMPLES_PER_TASK,
        output_file=OUTPUT_FILE
    )
    print(f"Sample generation complete. Results saved to: {{OUTPUT_FILE}}")


if __name__ == "__main__":
    # Execute main function when script is run directly
    main()