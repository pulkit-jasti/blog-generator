# graph.py
from langgraph.graph import StateGraph, END
from state import BlogAgentState
from nodes.crawler import crawl_and_extract
from nodes.analyst import analyze_business
from nodes.strategist import generate_topics
from nodes.validator import validate_topics
from nodes.writer import write_briefs
from nodes.reviewer import review_briefs


def route_after_validation(state: BlogAgentState) -> str:
    approved = state.get("approved_topics")
    retry_count = state.get("retry_count", 0)
    MAX_RETRIES = 2

    if approved and len(approved) >= 7:
        print(f"\n[Router] Validation passed -> moving to writer")
        return "writer"

    if retry_count >= MAX_RETRIES:
        print(f"\n[Router] Max retries ({MAX_RETRIES}) reached -> saving best available")
        return "writer"      # write whatever we have rather than saving nothing

    print(f"\n[Router] Validation failed -> retrying strategist (attempt {retry_count + 1})")
    return "strategist"


def build_graph():
    graph = StateGraph(BlogAgentState)

    # ── Nodes ─────────────────────────────────────────────────────
    graph.add_node("crawl", crawl_and_extract)
    graph.add_node("analyst", analyze_business)
    graph.add_node("strategist", generate_topics)
    graph.add_node("validator", validate_topics)
    graph.add_node("writer", write_briefs)
    graph.add_node("reviewer", review_briefs)

    # ── Fixed edges ────────────────────────────────────────────────
    graph.set_entry_point("crawl")
    graph.add_edge("crawl", "analyst")
    graph.add_edge("analyst", "strategist")
    graph.add_edge("strategist", "validator")
    graph.add_edge("writer", "reviewer")       # writer feeds reviewer
    graph.add_edge("reviewer", END)            # reviewer is the last node

    # ── Conditional edge ───────────────────────────────────────────
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "strategist": "strategist",
            "writer": "writer",
        }
    )

    return graph.compile()