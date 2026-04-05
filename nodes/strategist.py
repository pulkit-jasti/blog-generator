# nodes/strategist.py
#
# CONCEPT: Agent chaining in action
# This node reads business_summary (written by analyst node).
# It never touches raw_pages — it trusts the analyst's work.
# This is specialization: each agent has one job and one input source.
#
# CONCEPT: Validating a list with Pydantic
# Instead of validating one object, we validate each item in a list.
# If any item fails validation we catch it and still save what we can.

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError
from typing import List
from state import BlogAgentState
from prompts.strategist_prompt import (
    STRATEGIST_SYSTEM_PROMPT,
    build_strategist_user_prompt
)


# ── Pydantic model for one topic ──────────────────────────────────────────────

class TopicCandidate(BaseModel):
    title: str
    search_intent: str
    target_audience: str
    content_angle: str
    target_landing_page: str
    primary_keyword: str
    score_reason: str


# ── The node function ─────────────────────────────────────────────────────────

def generate_topics(state: BlogAgentState) -> dict:
    """
    Reads business_summary from state.
    Calls Gemini to generate 10 blog topic candidates.
    Returns topic_candidates to be merged into state.

    What this node reads:  state["business_summary"]
    What this node writes: state["topic_candidates"]
    """

    print(f"\n[Strategist] Generating blog topics...")

    # Guard: if analyst failed, skip
    if not state["business_summary"]:
        print("[Strategist] No business summary found. Skipping.")
        return {"topic_candidates": None}

    try:
        # ── Step 1: Initialize LLM ───────────────────────────────────────────
        # Same pattern as analyst — swap model name here to change LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.4,   # slight creativity for topic variety
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # ── Step 2: Build messages ───────────────────────────────────────────
        # CONCEPT: Notice we pass business_summary, not raw_pages
        # The strategist only knows what the analyst told it.
        # Clean handoff between agents through shared state.
        user_prompt = build_strategist_user_prompt(state["business_summary"])

        messages = [
            SystemMessage(content=STRATEGIST_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        # ── Step 3: Call Gemini ──────────────────────────────────────────────
        print("[Strategist] Calling Gemini...")
        response = llm.invoke(messages)

        # ── Step 4: Extract text (same fix as analyst) ───────────────────────
        # CONCEPT: Reusable pattern
        # Every node that calls Gemini needs this same extraction.
        # In a larger project you'd put this in a utils.py helper.
        if isinstance(response.content, list):
            raw_text = response.content[0].get("text", "")
        else:
            raw_text = response.content

        print(f"[Strategist] Got response ({len(raw_text)} chars)")

        # ── Step 5: Clean and parse JSON ─────────────────────────────────────
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]

        parsed = json.loads(clean_text)

        # Gemini should return a list — but double check
        if not isinstance(parsed, list):
            print("[Strategist] Expected a list, got something else.")
            return {"topic_candidates": None}

        # ── Step 6: Validate each topic with Pydantic ────────────────────────
        # CONCEPT: Validating a list
        # We loop through each item and validate individually.
        # Bad items are skipped, good items are kept.
        # This way one bad topic doesn't kill all 10.
        valid_topics = []
        for i, item in enumerate(parsed):
            try:
                topic = TopicCandidate(**item)
                valid_topics.append(topic.model_dump())
            except ValidationError as e:
                print(f"[Strategist] Topic {i+1} failed validation — skipping. {e}")

        print(f"[Strategist] OK {len(valid_topics)} valid topics generated")
        for i, t in enumerate(valid_topics):
            print(f"  {i+1}. {t['title']}")

        return {"topic_candidates": valid_topics}

    except json.JSONDecodeError as e:
        print(f"[Strategist] JSON parse failed: {e}")
        print(f"[Strategist] Raw text was: {raw_text[:300]}")
        return {"topic_candidates": None}

    except Exception as e:
        import traceback
        print(f"[Strategist] Unexpected error: {e}")
        print(traceback.format_exc())
        return {"topic_candidates": None}