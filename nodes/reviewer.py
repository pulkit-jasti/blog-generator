# nodes/reviewer.py
#
# CONCEPT: The final quality gate
# The reviewer is the last agent before saving.
# Unlike the validator (which rejects bad topics),
# the reviewer FIXES and REFINES — it does not reject.
# Its job is to make good briefs great, not to gatekeep.
#
# CONCEPT: Sending a large payload to Gemini
# We send ALL briefs at once to check consistency across them.
# This is why we cap the reviewer to one call — if we called
# per brief we would lose the cross-brief consistency check.

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from state import BlogAgentState
from prompts.reviewer_prompt import REVIEWER_SYSTEM_PROMPT, build_reviewer_user_prompt


# ── Pydantic model ────────────────────────────────────────────────────────────

class OutlineSection(BaseModel):
    section_title: str
    key_points: List[str]
    word_count_target: int


class FinalBrief(BaseModel):
    title: str
    meta_description: str
    introduction: str
    outline: List[OutlineSection]
    conclusion: str
    cta: str
    internal_link_suggestion: str
    faq_ideas: List[str]
    primary_keyword: str
    secondary_keywords: List[str]
    estimated_word_count: int
    editor_notes: str


# ── The node function ─────────────────────────────────────────────────────────

def review_briefs(state: BlogAgentState) -> dict:
    """
    Reads blog_briefs from state.
    Calls Gemini once to review and refine all briefs.
    Returns final_briefs to be merged into state.

    What this node reads:  state["blog_briefs"], state["business_summary"]
    What this node writes: state["final_briefs"]
    """

    briefs = state.get("blog_briefs") or []
    print(f"\n[Reviewer] Reviewing {len(briefs)} blog briefs...")

    if not briefs:
        print("[Reviewer] No briefs to review. Skipping.")
        return {"final_briefs": None}

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        user_prompt = build_reviewer_user_prompt(
            blog_briefs=briefs,
            business_summary=state["business_summary"]
        )

        messages = [
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        print("[Reviewer] Calling Gemini...")
        response = llm.invoke(messages)

        # Extract text
        if isinstance(response.content, list):
            raw_text = response.content[0].get("text", "")
        else:
            raw_text = response.content

        print(f"[Reviewer] Got response ({len(raw_text)} chars)")

        # Clean and parse
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]

        parsed = json.loads(clean_text)

        # Gemini should return a list of refined briefs
        if not isinstance(parsed, list):
            print("[Reviewer] Expected a list. Saving unreviewed briefs.")
            return {"final_briefs": briefs}

        # Validate each reviewed brief
        final_briefs = []
        for i, item in enumerate(parsed):
            try:
                brief = FinalBrief(**item)
                final_briefs.append(brief.model_dump())
                print(f"  OK  {brief.title[:60]}...")
                if brief.editor_notes and brief.editor_notes.upper() != "OK":
                    print(f"      Notes: {brief.editor_notes[:80]}")
            except ValidationError as e:
                print(f"  FAIL  Brief {i+1} failed validation — keeping original")
                # Fall back to original unreviewed brief
                if i < len(briefs):
                    final_briefs.append(briefs[i])

        print(f"\n[Reviewer] Done. {len(final_briefs)} final briefs ready.")
        return {"final_briefs": final_briefs}

    except json.JSONDecodeError as e:
        print(f"[Reviewer] JSON parse failed: {e}")
        print("[Reviewer] Saving unreviewed briefs as final.")
        return {"final_briefs": briefs}

    except Exception as e:
        import traceback
        print(f"[Reviewer] Unexpected error: {e}")
        print(traceback.format_exc())
        return {"final_briefs": briefs}