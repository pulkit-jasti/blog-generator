# nodes/validator.py
#
# CONCEPT: Conditional routing node
# This node does two things:
# 1. Validates the topics (like any other agent)
# 2. Sets up the information the graph needs to decide what runs next
#
# The actual routing decision happens in the edge function in graph.py.
# This node just writes "validation_passed" into state.
# The graph reads that flag and routes accordingly.
#
# CONCEPT: Retry counter
# We increment retry_count every time validation fails.
# The edge function checks retry_count to enforce the max retry limit.
# Without this, a failing strategist could loop forever.

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from state import BlogAgentState
from prompts.validator_prompt import (
    VALIDATOR_SYSTEM_PROMPT,
    build_validator_user_prompt
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ApprovedTopic(BaseModel):
    title: str
    search_intent: str
    target_audience: str
    content_angle: str
    target_landing_page: str
    primary_keyword: str
    score_reason: str


class RejectedTopic(BaseModel):
    title: str
    reason: str


class ValidationResult(BaseModel):
    approved_topics: List[ApprovedTopic]
    rejected_topics: List[RejectedTopic]
    validation_passed: bool
    feedback: str


# ── The node function ─────────────────────────────────────────────────────────

def validate_topics(state: BlogAgentState) -> dict:
    """
    Reads topic_candidates and business_summary from state.
    Calls Gemini to validate each topic against strict rules.
    Returns approved_topics, rejection_reasons, and validation_passed.

    What this node reads:  state["topic_candidates"], state["business_summary"]
    What this node writes: state["approved_topics"], state["rejection_reasons"],
                           state["retry_count"]
    """

    print(f"\n[Validator] Validating {len(state.get('topic_candidates') or [])} topics...")
    print(f"[Validator] Retry count so far: {state['retry_count']}")

    # Guard: nothing to validate
    if not state["topic_candidates"]:
        print("[Validator] No topics to validate. Skipping.")
        return {
            "approved_topics": None,
            "rejection_reasons": ["No topics were generated"],
            "validation_passed": False
        }

    try:
        # ── Step 1: Initialize LLM ───────────────────────────────────────────
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,        # no creativity for a validation task
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # ── Step 2: Build messages ───────────────────────────────────────────
        # CONCEPT: Passing retry_count to the prompt
        # The validator knows how many times we have retried.
        # It uses this to adjust its strictness.
        user_prompt = build_validator_user_prompt(
            business_summary=state["business_summary"],
            topic_candidates=state["topic_candidates"],
            retry_count=state["retry_count"]
        )

        messages = [
            SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        # ── Step 3: Call Gemini ──────────────────────────────────────────────
        print("[Validator] Calling Gemini...")
        response = llm.invoke(messages)

        # ── Step 4: Extract text ─────────────────────────────────────────────
        if isinstance(response.content, list):
            raw_text = response.content[0].get("text", "")
        else:
            raw_text = response.content

        # ── Step 5: Clean and parse JSON ─────────────────────────────────────
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]

        parsed = json.loads(clean_text)

        # ── Step 6: Validate with Pydantic ───────────────────────────────────
        result = ValidationResult(**parsed)

        # ── Step 7: Print results ─────────────────────────────────────────────
        approved = [t.model_dump() for t in result.approved_topics]
        rejected = result.rejected_topics

        print(f"[Validator] Approved : {len(approved)} topics")
        print(f"[Validator] Rejected : {len(rejected)} topics")
        print(f"[Validator] Passed   : {result.validation_passed}")

        if approved:
            print("[Validator] Approved topics:")
            for t in approved:
                print(f"  OK  {t['title']}")

        if rejected:
            print("[Validator] Rejected topics:")
            for t in rejected:
                print(f"  FAIL  {t.title}")
                print(f"        Reason: {t.reason}")

        if not result.validation_passed:
            print(f"\n[Validator] Feedback for strategist:\n  {result.feedback}")

        # ── Step 8: Return state updates ─────────────────────────────────────
        # CONCEPT: What we return here matters for routing
        # The graph edge function will read "validation_passed" from state
        # to decide whether to go to the writer or back to the strategist.
        return {
            "approved_topics": approved,
            "rejection_reasons": [t.reason for t in rejected],
            "topic_candidates": approved,   # update candidates to approved list
            "retry_count": state["retry_count"] + 1 if not result.validation_passed else state["retry_count"],
        }

    except json.JSONDecodeError as e:
        print(f"[Validator] JSON parse failed: {e}")
        return {
            "approved_topics": None,
            "rejection_reasons": [f"Validator JSON error: {e}"],
            "retry_count": state["retry_count"] + 1
        }

    except Exception as e:
        import traceback
        print(f"[Validator] Unexpected error: {e}")
        print(traceback.format_exc())
        return {
            "approved_topics": None,
            "rejection_reasons": [str(e)],
            "retry_count": state["retry_count"] + 1
        }