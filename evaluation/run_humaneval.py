"""Run HumanEval evaluation on the code agent.

This script evaluates the agent's ability to complete Python functions
by running them against unit tests.

Usage:
    python -m evaluation.run_humaneval --model openrouter/meta-llama/llama-3.1-8b-instruct
    python -m evaluation.run_humaneval --model openrouter/anthropic/claude-3.5-sonnet --problems 0,1,2
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from evaluation.humaneval_problems import HUMANEVAL_PROBLEMS, get_problem


def execute_code_with_tests(code: str, test_code: str, entry_point: str) -> dict:
    """Execute generated code with tests and return results."""
    full_code = code + "\n" + test_code

    try:
        exec_globals = {}
        exec(full_code, exec_globals)
        return {"passed": True, "error": None}
    except AssertionError as e:
        return {"passed": False, "error": f"AssertionError: {e}"}
    except Exception as e:
        return {"passed": False, "error": f"{type(e).__name__}: {e}"}


def extract_code_from_response(response: str, prompt: str) -> str:
    """Extract the generated code from agent response."""
    import re

    # Try to find Python code block
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            code = matches[-1].strip()
            # If code doesn't include the function def, prepend the prompt
            if "def " not in code:
                return prompt + code
            return code

    # If no code block, try to find function definition
    if "def " in response:
        # Find from def to the end or next major section
        start = response.find("def ")
        code = response[start:]
        # Clean up
        lines = []
        for line in code.split("\n"):
            if line.strip().startswith("#") or line.strip() == "" or line.startswith(" ") or line.startswith("\t") or line.startswith("def "):
                lines.append(line)
            elif lines and not line.startswith("def "):
                break
        return "\n".join(lines)

    return prompt + response


def evaluate_single_problem(
    problem: dict,
    model_name: str,
    verbose: bool = False,
) -> dict:
    """Evaluate agent on a single HumanEval problem."""
    from code_agent.model.litellm import LiteLLMModel

    task_id = problem["task_id"]
    prompt = problem["prompt"]
    test_code = problem["test"]
    entry_point = problem["entry_point"]

    # Create prompt for the agent
    task = f"""Complete the following Python function. Only provide the complete function implementation.

{prompt}

Provide ONLY the complete Python function, nothing else. Do not include example usage or tests."""

    # Initialize model
    model = LiteLLMModel(model_name=model_name, temperature=0.0)

    # Query model directly (not using agent loop for code completion)
    messages = [
        {"role": "system", "content": "You are a Python coding assistant. Complete the function exactly as specified. Return only the Python code, no explanations."},
        {"role": "user", "content": task}
    ]

    start_time = time.time()
    try:
        response = model.query(messages)
        generated = response["content"]
        elapsed = time.time() - start_time

        # Extract code
        code = extract_code_from_response(generated, prompt)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task_id}")
            print(f"{'='*60}")
            print(f"Generated code:\n{code}")
            print(f"{'='*60}")

        # Run tests
        result = execute_code_with_tests(code, test_code, entry_point)

        return {
            "task_id": task_id,
            "passed": result["passed"],
            "error": result["error"],
            "generated_code": code,
            "elapsed_time": elapsed,
            "model_stats": model.get_stats(),
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "passed": False,
            "error": f"Generation error: {e}",
            "generated_code": None,
            "elapsed_time": time.time() - start_time,
            "model_stats": None,
        }


def run_evaluation(
    model_name: str,
    problem_ids: list[int] = None,
    verbose: bool = False,
    output_file: str = None,
) -> dict:
    """Run evaluation on multiple problems."""

    if problem_ids is None:
        problems = HUMANEVAL_PROBLEMS
    else:
        problems = [HUMANEVAL_PROBLEMS[i] for i in problem_ids if i < len(HUMANEVAL_PROBLEMS)]

    print(f"\nHumanEval Evaluation")
    print(f"Model: {model_name}")
    print(f"Problems: {len(problems)}")
    print("=" * 60)

    results = []
    passed = 0
    total = len(problems)

    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        print(f"\n[{i+1}/{total}] Evaluating {task_id}...", end=" ", flush=True)

        result = evaluate_single_problem(problem, model_name, verbose)
        results.append(result)

        if result["passed"]:
            passed += 1
            print("PASSED")
        else:
            print(f"FAILED: {result['error'][:50]}...")

    # Summary
    pass_rate = passed / total if total > 0 else 0

    summary = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "total_problems": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": pass_rate,
        "results": results,
    }

    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"Model: {model_name}")
    print(f"Passed: {passed}/{total} ({pass_rate*100:.1f}%)")
    print(f"Failed: {total - passed}/{total}")

    # Show failed problems
    failed_problems = [r for r in results if not r["passed"]]
    if failed_problems:
        print(f"\nFailed problems:")
        for r in failed_problems:
            print(f"  - {r['task_id']}: {r['error'][:60]}...")

    # Save results
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run HumanEval evaluation")
    parser.add_argument(
        "--model",
        default="openrouter/meta-llama/llama-3.1-8b-instruct",
        help="Model to evaluate"
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="Comma-separated problem indices (e.g., 0,1,2). Default: all"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show generated code"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results JSON"
    )

    args = parser.parse_args()

    problem_ids = None
    if args.problems:
        problem_ids = [int(x.strip()) for x in args.problems.split(",")]

    output_file = args.output
    if output_file is None:
        model_short = args.model.split("/")[-1]
        output_file = f"evaluation/results/humaneval_{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    run_evaluation(
        model_name=args.model,
        problem_ids=problem_ids,
        verbose=args.verbose,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
