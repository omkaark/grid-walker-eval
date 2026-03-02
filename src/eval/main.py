import argparse
import asyncio
import json
import os
from dotenv import load_dotenv

from .harness import run_eval


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Grid Walker Eval")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api-key", type=str, default=None, help="API key (overrides OPENROUTER_API_KEY)")
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated values like 0,1,2")
    parser.add_argument("--grid-size", type=int, default=8, help="Grid size")
    parser.add_argument("--blocks", type=int, default=3, help="Number of obstacle blocks")
    parser.add_argument("--max-turns", type=int, default=50, help="Max turns per episode")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--log-images", type=str, default=None, metavar="RUN_NAME",
                        help="Save screenshots to outputs/RUN_NAME/seed_X/step_N.png")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: set --api-key or OPENROUTER_API_KEY in environment/.env")
        return 1

    seeds = [s.strip() for s in args.seeds.split(",")]

    results = asyncio.run(run_eval(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        seeds=seeds,
        grid_size=args.grid_size,
        blocks=args.blocks,
        max_turns=args.max_turns,
        verbose=args.verbose,
        log_images=args.log_images
    ))

    output = results.to_dict()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))

    print(f"\nSuccess rate: {results.success_rate:.1%}")
    return 0


if __name__ == "__main__":
    exit(main())
