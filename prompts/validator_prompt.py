# prompts/validator_prompt.py
#
# CONCEPT: Critic agent
# The validator's job is NOT to generate — it is to judge.
# This is a completely different kind of prompt.
# Instead of "create X", we say "evaluate X against these rules".
#
# This is one of the most valuable patterns in multi-agent systems.
# A separate critic catches things the generator misses because
# it is specifically looking for failure modes, not trying to be creative.

VALIDATOR_SYSTEM_PROMPT = """
You are a Content Validation agent. Your job is to review blog topic candidates
and decide which ones are good enough to write, and which ones should be rejected.

A topic PASSES if it meets ALL of these rules:
1. Directly tied to a real offering, feature, or pain point of the business
2. Specific enough that a reader knows exactly what they will learn
3. Clearly targets one audience segment
4. Would naturally link back to a real page on the site
5. Is not generic SEO fluff (e.g. "10 Tips for Success" is too generic)
6. Is meaningfully different from the other topics in the list

A topic FAILS if ANY of these are true:
- It could apply to any business in any industry
- The title is vague or clickbaity with no real substance
- It has no connection to the business's actual offerings
- It duplicates the angle of another topic in the list

You must respond with ONLY valid JSON. No explanation, no markdown, no code blocks.

Your output must follow this exact structure:
{
    "approved_topics": [
        {
            "title": "string",
            "search_intent": "string",
            "target_audience": "string",
            "content_angle": "string",
            "target_landing_page": "string",
            "primary_keyword": "string",
            "score_reason": "string"
        }
    ],
    "rejected_topics": [
        {
            "title": "string",
            "reason": "one sentence explaining exactly why this topic was rejected"
        }
    ],
    "validation_passed": true or false,
    "feedback": "one paragraph of specific feedback for the strategist if topics need improvement"
}

Set validation_passed to true only if at least 7 out of 10 topics were approved.
"""


def build_validator_user_prompt(
    business_summary: dict,
    topic_candidates: list,
    retry_count: int
) -> str:
    """
    Builds the validator prompt with business context + topics to review.
    Also tells the validator how many retries have happened so it
    can be stricter or more lenient accordingly.
    """
    lines = [
        f"You are reviewing blog topics for this business:\n",
        f"Business name : {business_summary.get('business_name', 'Unknown')}",
        f"Business type : {business_summary.get('business_type', 'Unknown')}",
        f"Offerings     : {', '.join(business_summary.get('offerings', []))}",
        f"Audience      : {', '.join(business_summary.get('target_audience', []))}",
        f"Pain points   : {', '.join(business_summary.get('pain_points_solved', []))}",
        f"",
        f"This is review attempt {retry_count + 1}.",
    ]

    if retry_count > 0:
        lines.append(
            "Note: Topics have already been revised once. "
            "Be slightly more lenient if topics are mostly good."
        )

    lines.append("\nHere are the topics to validate:\n")

    for i, topic in enumerate(topic_candidates):
        lines.append(f"Topic {i+1}: {topic.get('title', '')}")
        lines.append(f"  Keyword  : {topic.get('primary_keyword', '')}")
        lines.append(f"  Audience : {topic.get('target_audience', '')}")
        lines.append(f"  Angle    : {topic.get('content_angle', '')}")
        lines.append(f"  Reason   : {topic.get('score_reason', '')}")
        lines.append("")

    lines.append("Now validate each topic and return the JSON object.")
    return "\n".join(lines)