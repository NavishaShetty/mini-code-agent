"""Main CLI entry point for Code Agent.

Usage:
    # Run agent with a task
    python -m code_agent.main --task "List all Python files"

    # Different execution modes
    python -m code_agent.main --task "..." --mode confirm  # Default
    python -m code_agent.main --task "..." --mode yolo     # Full autonomy
    python -m code_agent.main --task "..." --mode human    # User provides commands

    # Different models
    python -m code_agent.main --task "..." --model claude-sonnet-4-20250514
    python -m code_agent.main --task "..." --model gpt-4o
"""

import argparse
import sys

from code_agent.agent.interactive import ExecutionMode, InteractiveAgent
from code_agent.environment.local import LocalEnvironment
from code_agent.model.litellm import LiteLLMModel


def main():
    parser = argparse.ArgumentParser(
        description="Code Agent - ReAct pattern demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m code_agent.main --task "List all Python files in src/"
  python -m code_agent.main --task "Find the main function" --mode yolo
  python -m code_agent.main --task "Refactor this code" --mode confirm
        """,
    )

    parser.add_argument(
        "--task", "-t",
        required=True,
        help="The task for the agent to complete",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["human", "confirm", "yolo"],
        default="confirm",
        help="Execution mode (default: confirm)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="LLM model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=20,
        help="Maximum number of steps (default: 20)",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=1.0,
        help="Maximum cost in USD (default: 1.0)",
    )
    parser.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="Disable auto-approval of safe commands in confirm mode",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Initialize components
    print(f"🚀 Initializing agent...")
    print(f"   Model: {args.model}")
    print(f"   Mode: {args.mode}")
    print(f"   Limits: {args.step_limit} steps, ${args.cost_limit}")

    try:
        model = LiteLLMModel(model_name=args.model)
        env = LocalEnvironment()
        agent = InteractiveAgent(
            model,
            env,
            mode=ExecutionMode(args.mode),
            auto_approve_safe=not args.no_auto_approve,
            step_limit=args.step_limit,
            cost_limit=args.cost_limit,
        )

        # Run the agent
        status, message = agent.run(args.task)

        # Print results
        print("\n" + "=" * 50)
        print(f"Status: {status}")
        print(f"Message: {message[:500]}..." if len(message) > 500 else f"Message: {message}")

        # Print stats
        stats = model.get_stats()
        print(f"\nStats:")
        print(f"  API calls: {stats.get('n_calls', 0)}")
        print(f"  Total cost: ${stats.get('total_cost', 0):.4f}")
        print(f"  Tokens: {stats.get('total_tokens', 0)}")

        return 0 if status == "Submitted" else 1

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
