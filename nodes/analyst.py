# nodes/analyst.py
#
# CONCEPT: LLM inside a node
# A node is still just a function. The only difference from the crawler
# is that instead of doing Python work, we call an LLM and parse the result.
#
# CONCEPT: Structured output
# LLMs return text. We need a dict. So we:
#   1. Tell the LLM in the prompt to return only JSON
#   2. Parse the response with json.loads()
#   3. Validate the shape with Pydantic
# If parsing fails, we store the error in state instead of crashing.

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError
from typing import List
from state import BlogAgentState
from prompts.analyst_prompt import ANALYST_SYSTEM_PROMPT, build_analyst_user_prompt


# ── Pydantic model ────────────────────────────────────────────────────────────
# CONCEPT: Pydantic validation
# Pydantic lets you define the exact shape you expect from the LLM.
# If the LLM returns something unexpected, Pydantic raises a clear error
# instead of silently passing bad data to the next agent.

class BusinessSummary(BaseModel):
    business_name: str
    business_type: str
    description: str
    offerings: List[str]
    target_audience: List[str]
    pain_points_solved: List[str]
    key_benefits: List[str]
    tone: str
    missing_info: str


# ── The node function ─────────────────────────────────────────────────────────

def analyze_business(state: BlogAgentState) -> dict:
    """
    Reads raw_pages from state.
    Calls Gemini to extract a business summary.
    Returns business_summary to be merged into state.

    What this node reads:  state["raw_pages"]
    What this node writes: state["business_summary"]
    """

    print(f"\n[Analyst] Analyzing business from {len(state['raw_pages'])} pages...")

    # Guard: if crawler failed, don't even try
    if not state["raw_pages"]:
        print("[Analyst] No pages found. Skipping.")
        return {
            "business_summary": None,
        }

    try:
        # ── Step 1: Initialize the LLM ───────────────────────────────────────
        # CONCEPT: ChatGoogleGenerativeAI
        # This is LangChain's wrapper around Gemini.
        # temperature=0 means no creativity — we want consistent, factual output.
        # Changing to a different LLM later = just swap this one line.
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # ── Step 2: Build the messages ───────────────────────────────────────
        # CONCEPT: SystemMessage + HumanMessage
        # This is how you talk to a chat model.
        # SystemMessage = instructions (who the LLM is, what format to use)
        # HumanMessage  = the actual input for this run
        user_prompt = build_analyst_user_prompt(state["raw_pages"])

        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        # ── Step 3: Call the LLM ─────────────────────────────────────────────
        print("[Analyst] Calling Gemini...")
        response = llm.invoke(messages)

        # Newer langchain-google-genai returns a list of content blocks
        # We extract the text from the first block
        # Gemini returns a list of content blocks like:
        # [{'type': 'text', 'text': '...actual json...', 'extras': {...}}]
        # We need to dig into the first block and grab the 'text' key
        if isinstance(response.content, list):
            raw_text = response.content[0].get("text", "")
        else:
            raw_text = response.content

        print(f"[Analyst] Extracted text preview: {raw_text[:100]}")
        # ── Step 4: Parse JSON ───────────────────────────────────────────────
        # CONCEPT: Cleaning LLM output
        # Even with instructions, some LLMs wrap JSON in ```json ... ```
        # This strips that safely before parsing.
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]

        parsed = json.loads(clean_text)

        # ── Step 5: Validate with Pydantic ───────────────────────────────────
        summary = BusinessSummary(**parsed)

        print(f"[Analyst] OK Business identified: {summary.business_name}")
        print(f"[Analyst] Type: {summary.business_type}")
        print(f"[Analyst] Offerings: {summary.offerings[:3]}")

        # Return ONLY what this node changed
        return {
            "business_summary": summary.model_dump()
        }

    except json.JSONDecodeError as e:
        print(f"[Analyst] JSON parse failed: {e}")
        print(f"[Analyst] Raw response was: {raw_text[:300]}")
        return {"business_summary": None}

    except ValidationError as e:
        print(f"[Analyst] Pydantic validation failed: {e}")
        return {"business_summary": None}

    except Exception as e:
        import traceback
        print(f"[Analyst] Unexpected error: {e}")
        print(traceback.format_exc())   # ← prints the full error chain
        return {"business_summary": None}