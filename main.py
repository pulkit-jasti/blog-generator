# main.py
import json
from dotenv import load_dotenv
from graph import build_graph

load_dotenv()


def run(url: str):
    graph = build_graph()

    initial_state = {
        "url": url,
        "raw_pages": [],
        "crawl_error": None,
        "business_summary": None,
        "topic_candidates": None,
        "approved_topics": None,
        "rejection_reasons": None,
        "retry_count": 0,
        "validation_passed": None,
        "blog_briefs": None,
        "final_briefs": None,
    }

    print(f"\n{'='*50}")
    print(f"Running blog agent on: {url}")
    print(f"{'='*50}")

    final_state = graph.invoke(initial_state)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Pages crawled   : {len(final_state['raw_pages'])}")

    summary = final_state.get("business_summary")
    if summary:
        print(f"Business        : {summary['business_name']}")

    approved = final_state.get("approved_topics") or []
    print(f"Topics approved : {len(approved)}")
    print(f"Retry count     : {final_state.get('retry_count', 0)}")

    final_briefs = final_state.get("final_briefs") or []
    print(f"Final briefs    : {len(final_briefs)}")

    # ── Print each final brief ────────────────────────────────────
    if final_briefs:
        print(f"\n--- BLOG BRIEFS ---\n")
        for i, brief in enumerate(final_briefs):
            print(f"Brief {i+1}: {brief['title']}")
            print(f"  Keyword    : {brief['primary_keyword']}")
            print(f"  Word count : {brief['estimated_word_count']}")
            print(f"  Sections   : {len(brief.get('outline', []))}")
            print(f"  CTA        : {brief['cta'][:70]}")
            if brief.get('editor_notes') and brief['editor_notes'].upper() != "OK":
                print(f"  Editor     : {brief['editor_notes'][:80]}")
            print()

    # ── Save full output ──────────────────────────────────────────
    with open("output/final_briefs.json", "w", encoding="utf-8") as f:
        json.dump(final_state["final_briefs"], f, indent=2, ensure_ascii=False)

    with open("output/full_run.json", "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=2, default=str, ensure_ascii=False)

    print(f"Saved to output/final_briefs.json")
    print(f"Saved to output/full_run.json")


if __name__ == "__main__":
    run("https://menudoodle.com/")