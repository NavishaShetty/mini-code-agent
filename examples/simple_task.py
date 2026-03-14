"""Simple example of running the code agent.

This demonstrates the basic usage of the agent.

Run with:
    python examples/simple_task.py
"""

from code_agent import InteractiveAgent, ExecutionMode, LocalEnvironment, LiteLLMModel


def main():
    # Initialize components
    print("Initializing agent...")

    model = LiteLLMModel(model_name="claude-sonnet-4-20250514")
    env = LocalEnvironment()

    # Create agent in confirm mode (safest for demos)
    agent = InteractiveAgent(
        model,
        env,
        mode=ExecutionMode.CONFIRM,
        step_limit=10,
        cost_limit=0.50,
    )

    # Run a simple task
    task = "List all Python files in the current directory and count how many there are"

    print(f"\n📋 Task: {task}\n")
    status, message = agent.run(task)

    # Show results
    print("\n" + "=" * 50)
    print(f"Status: {status}")
    print(f"Stats: {model.get_stats()}")


if __name__ == "__main__":
    main()
