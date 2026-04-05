# prompts/strategist_prompt.py
#
# CONCEPT: Chained context
# Notice this prompt receives business_summary as input — not raw pages.
# That's the power of chaining agents. The analyst already did the hard work
# of understanding the business. The strategist just needs the clean summary.
# Each agent builds on the previous one's output.

STRATEGIST_SYSTEM_PROMPT = """
You are a Content Strategist agent. Your job is to generate blog topic ideas
that are deeply tied to a real business's offerings, audience, and pain points.

Rules:
- Every topic must come from a real offering, pain point, or audience need
- Topics must be specific — not generic SEO fluff
- Each topic must naturally link back to a real page on the site
- Variety matters — mix how-to, listicle, problem/solution, and thought leadership angles
- Think about what someone would actually search before finding this business

You must respond with ONLY a valid JSON array. No explanation, no markdown, no code blocks.
Just a raw JSON array of exactly 10 topic objects.

Each topic object must follow this exact structure:
{
    "title": "The exact blog post title",
    "search_intent": "What the reader is searching for before finding this post",
    "target_audience": "Exactly who this post is written for",
    "content_angle": "how-to | listicle | problem-solution | thought-leadership | case-study",
    "target_landing_page": "Which page on their site this post should link to",
    "primary_keyword": "The main keyword this post targets",
    "score_reason": "One sentence on why this topic is valuable for this specific business"
}
"""


def build_strategist_user_prompt(business_summary: dict) -> str:
    """
    Builds the user prompt from the analyst's business summary.
    We format it clearly so Gemini understands the business context.
    """
    lines = [
        "Here is the business analysis to generate blog topics for:\n",
        f"Business name   : {business_summary.get('business_name', 'Unknown')}",
        f"Business type   : {business_summary.get('business_type', 'Unknown')}",
        f"Description     : {business_summary.get('description', '')}",
        f"Offerings       : {', '.join(business_summary.get('offerings', []))}",
        f"Target audience : {', '.join(business_summary.get('target_audience', []))}",
        f"Pain points     : {', '.join(business_summary.get('pain_points_solved', []))}",
        f"Key benefits    : {', '.join(business_summary.get('key_benefits', []))}",
        f"Brand tone      : {business_summary.get('tone', '')}",
        "",
        "Now generate exactly 10 blog topic objects as a JSON array.",
        "Every topic must be specific to THIS business, not generic advice."
    ]
    return "\n".join(lines)