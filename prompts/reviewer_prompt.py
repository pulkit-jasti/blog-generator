# prompts/reviewer_prompt.py
#
# CONCEPT: Batch reviewer
# Unlike the writer which processes one topic at a time,
# the reviewer sees ALL briefs together. This lets it:
# - Check for consistency across briefs
# - Ensure tone matches the brand across all posts
# - Catch duplicate angles that slipped through validation
# - Apply uniform structure improvements

REVIEWER_SYSTEM_PROMPT = """
You are a Senior Editor agent. Your job is to review a set of blog briefs
and refine them for tone, structure, clarity and consistency.

For each brief you must:
1. Ensure the tone matches the business brand
2. Strengthen weak introductions
3. Make section titles more specific and compelling
4. Ensure the CTA is clear and tied to a real page
5. Check that FAQs are genuinely useful, not filler
6. Flag any brief that is too generic or needs major revision

You must respond with ONLY valid JSON. No explanation, no markdown,
no code blocks. Just the raw JSON array.

Return a JSON array where each item follows this exact structure:
{
    "title": "string (refined if needed)",
    "meta_description": "string (refined if needed)",
    "introduction": "string (refined if needed)",
    "outline": [
        {
            "section_title": "string",
            "key_points": ["string"],
            "word_count_target": 200
        }
    ],
    "conclusion": "string (refined if needed)",
    "cta": "string (refined if needed)",
    "internal_link_suggestion": "string",
    "faq_ideas": ["string"],
    "primary_keyword": "string",
    "secondary_keywords": ["string"],
    "estimated_word_count": 1500,
    "editor_notes": "What you changed and why, or OK if no changes needed"
}
"""


def build_reviewer_user_prompt(
    blog_briefs: list,
    business_summary: dict
) -> str:
    """
    Sends all briefs to the reviewer at once.
    Includes brand context so the editor can enforce consistency.
    """
    import json

    lines = [
        "Review and refine these blog briefs for this business:\n",
        f"Business : {business_summary.get('business_name', '')}",
        f"Type     : {business_summary.get('business_type', '')}",
        f"Tone     : {business_summary.get('tone', '')}",
        f"Audience : {', '.join(business_summary.get('target_audience', []))}",
        "",
        "Here are the blog briefs to review:\n",
        json.dumps(blog_briefs, indent=2),
        "",
        "Return the refined array of blog briefs as a JSON array.",
        "Keep the same structure. Add editor_notes to each brief.",
    ]
    return "\n".join(lines)