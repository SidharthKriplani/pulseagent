"""
main.py — PulseAgent entry point
Usage:
  python3 main.py "how do I add a blog?"   # single query
  python3 main.py                           # interactive
  python3 main.py --legacy                  # use original single-agent graph
"""
import sys

def run_query(query: str, legacy: bool = False):
    if legacy:
        from src.agent.graph import run_agent
        result = run_agent(query)
    else:
        from src.agents.supervisor import run_supervisor
        result = run_supervisor(query)

    print(f"\nQuery: {query}")
    print(f"Route:  {result['route']}")
    print(f"Answer: {result['final_answer']}")
    if result.get("sub_queries"):
        print(f"Sub-queries: {result['sub_queries']}")
    if result.get("cited_ids"):
        print(f"Cited:  {result['cited_ids'][:3]}")

if __name__ == "__main__":
    legacy = "--legacy" in sys.argv
    args   = [a for a in sys.argv[1:] if a != "--legacy"]

    if args:
        run_query(" ".join(args), legacy=legacy)
    else:
        mode = "legacy single-agent" if legacy else "multi-agent supervisor"
        print(f"PulseAgent interactive [{mode}]. Ctrl+C to exit.\n")
        while True:
            try:
                q = input("Query > ").strip()
                if q: run_query(q, legacy=legacy)
            except (KeyboardInterrupt, EOFError):
                break
