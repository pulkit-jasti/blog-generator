# prompts/writer_prompt.py
#
# CONCEPT: One prompt, called multiple times
# The writer node loops over approved topics and calls this prompt
# once per topic. Same system prompt every time, different user prompt
# each time (because each topic is different).
#
# This is better than asking Gemini to write all 10 briefs at once because:
# - Each brief gets the model's full attention
# - If one fails, the others still succeed
# - Easier to validate each output individually

WRITER_SYSTEM_PROMPT = """
You are a Content Writer agent. Your job is to take a blog topic
and generate a detailed blog brief that a human writer can follow
to write a complete, high-quality blog post.

You must respond with ONLY valid JSON. No explanation, no markdown,
no code blocks. Just the raw JSON object.

Your output must follow this exact structure:
{
    "title": "The exact blog post title",
    "meta_description": "150-160 char SEO meta description",
    "introduction": "2-3 paragraph introduction that hooks the reader",
    "outline": [
        {
            "section_title": "H2 section heading",
            "key_points": ["point 1", "point 2", "point 3"],
            "word_count_target": 200
        }
    ],
    "conclusion": "1 paragraph conclusion with key takeaway",
    "cta": "The call to action at the end of the post",
    "internal_link_suggestion": "Which page on the site this post should link to",
    "faq_ideas": ["Question 1?", "Question 2?", "Question 3?"],
    "primary_keyword": "main keyword",
    "secondary_keywords": ["keyword 2", "keyword 3"],
    "estimated_word_count": 1500
}
"""


def build_writer_user_prompt(topic: dict, business_summary: dict) -> str:
    """
    Builds the prompt for one specific topic.
    We pass business context so the writer stays on-brand.
    """
    lines = [
        "Write a detailed blog brief for this topic:\n",
        f"Title          : {topic.get('title', '')}",
        f"Primary keyword: {topic.get('primary_keyword', '')}",
        f"Target audience: {topic.get('target_audience', '')}",
        f"Content angle  : {topic.get('content_angle', '')}",
        f"Search intent  : {topic.get('search_intent', '')}",
        f"Link target    : {topic.get('target_landing_page', '')}",
        f"Why valuable   : {topic.get('score_reason', '')}",
        "",
        "Business context:",
        f"  Name     : {business_summary.get('business_name', '')}",
        f"  Type     : {business_summary.get('business_type', '')}",
        f"  Tone     : {business_summary.get('tone', '')}",
        f"  Audience : {', '.join(business_summary.get('target_audience', []))}",
        "",
        "Now generate the full blog brief as a JSON object.",
        "Make the outline specific — not generic section titles.",
        "The introduction should hook the reader immediately.",
    ]
    return "\n".join(lines)