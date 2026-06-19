"""
main.py — PulseAgent entry point
Usage:
  python3 main.py "how do I add a blog?"   # single query
  python3 main.py                           # interactive
"""
import sys

def run_query(query: str):
    from src.agent.graph import run_agent
    print(f"\nQuery: {query}")
    print("Running agent...\n")
    result = run_agent(query)
    print(f"Route:  {result['route']}")
    print(f"Answer: {result['final_answer']}")
    if result["cited_ids"]:
        print(f"Cited:  {result['cited_ids'][:3]}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_query(" ".join(sys.argv[1:]))
    else:
        print("PulseAgent interactive. Ctrl+C to exit.\n")
        while True:
            try:
                q = input("Query > ").strip()
                if q: run_query(q)
            except (KeyboardInterrupt, EOFError):
                break
