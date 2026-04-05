# nodes/writer.py
#
# CONCEPT: Looping over a list inside a node
# This node calls Gemini once per approved topic.
# We loop in Python, not in the prompt.
# This gives us:
# - Better quality (full attention per topic)
# - Fault tolerance (failed topics don't kill the others)
# - Clear per-topic logging
#
# The tradeoff is more API calls. For 9 topics that is 9 Gemini calls.
# For a learning project this is fine. In production you would
# batch or parallelize these calls.

import json
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from state import BlogAgentState
from prompts.writer_prompt import WRITER_SYSTEM_PROMPT, build_writer_user_prompt


# ── Pydantic model ────────────────────────────────────────────────────────────

class OutlineSection(BaseModel):
    section_title: str
    key_points: List[str]
    word_count_target: int


class BlogBrief(BaseModel):
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


# ── Helper: call Gemini and parse one brief ───────────────────────────────────

def _generate_one_brief(
    llm: ChatGoogleGenerativeAI,
    topic: dict,
    business_summary: dict
) -> Optional[dict]:
    """
    Calls Gemini for one topic and returns a validated brief dict.
    Returns None if anything fails so the loop can continue.
    """
    try:
        user_prompt = build_writer_user_prompt(topic, business_summary)

        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = llm.invoke(messages)

        # Extract text from Gemini response
        if isinstance(response.content, list):
            raw_text = response.content[0].get("text", "")
        else:
            raw_text = response.content

        # Clean and parse JSON
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]

        parsed = json.loads(clean_text)

        # Validate with Pydantic
        brief = BlogBrief(**parsed)
        return brief.model_dump()

    except json.JSONDecodeError as e:
        print(f"    [Writer] JSON parse failed: {e}")
        return None

    except ValidationError as e:
        print(f"    [Writer] Pydantic validation failed: {e}")
        return None

    except Exception as e:
        print(f"    [Writer] Unexpected error: {e}")
        return None


# ── The node function ─────────────────────────────────────────────────────────

def write_briefs(state: BlogAgentState) -> dict:
    """
    Reads approved_topics from state.
    Calls Gemini once per topic to generate a blog brief.
    Returns blog_briefs to be merged into state.

    What this node reads:  state["approved_topics"], state["business_summary"]
    What this node writes: state["blog_briefs"]
    """

    approved = state.get("approved_topics") or []
    print(f"\n[Writer] Writing briefs for {len(approved)} approved topics...")

    if not approved:
        print("[Writer] No approved topics found. Skipping.")
        return {"blog_briefs": None}

    # Initialize LLM once, reuse for all topics
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    blog_briefs = []

    for i, topic in enumerate(approved):
        print(f"\n[Writer] Topic {i+1}/{len(approved)}: {topic.get('title', '')[:60]}...")

        brief = _generate_one_brief(llm, topic, state["business_summary"])

        if brief:
            blog_briefs.append(brief)
            print(f"  OK  Brief generated ({len(brief.get('outline', []))} sections)")
        else:
            print(f"  FAIL  Brief generation failed for this topic")

        # Small delay between calls to avoid rate limits
        if i < len(approved) - 1:
            time.sleep(1)

    print(f"\n[Writer] Done. {len(blog_briefs)}/{len(approved)} briefs generated.")

    return {"blog_briefs": blog_briefs}