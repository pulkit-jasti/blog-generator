# prompts/analyst_prompt.py
#
# CONCEPT: System prompt vs User prompt
# System prompt = the agent's job description. It tells the LLM WHO it is
# and HOW to behave. This never changes between runs.
#
# User prompt = the actual data for this specific run.
# We build it dynamically using the crawled page content.

ANALYST_SYSTEM_PROMPT = """
You are a Business Analyst agent. Your job is to read raw website content
and extract a structured understanding of the business.

You must respond with ONLY valid JSON. No explanation, no markdown, 
no code blocks. Just the raw JSON object.

Your output must follow this exact structure:
{
    "business_name": "string",
    "business_type": "string (e.g. SaaS, ecommerce, agency, etc.)",
    "description": "2-3 sentence summary of what the business does",
    "offerings": ["list", "of", "products", "or", "services"],
    "target_audience": ["list", "of", "audience", "segments"],
    "pain_points_solved": ["list", "of", "problems", "they", "solve"],
    "key_benefits": ["list", "of", "benefits", "they", "offer"],
    "tone": "string (e.g. professional, casual, technical, friendly)",
    "missing_info": "anything important you couldn't determine"
}
"""


def build_analyst_user_prompt(pages: list) -> str:
    """
    Builds the user prompt dynamically from crawled pages.
    We format each page clearly so the LLM can parse it easily.
    """
    lines = ["Here is the website content to analyze:\n"]

    for i, page in enumerate(pages[:5]):  # send max 5 pages to keep prompt lean
        lines.append(f"--- Page {i+1}: {page['url']} ---")
        lines.append(f"Title: {page['title']}")

        if page.get("headings"):
            lines.append(f"Headings: {', '.join(page['headings'][:8])}")

        if page.get("body"):
            lines.append(f"Content: {page['body'][:1500]}")  # cap per page

        if page.get("ctas"):
            lines.append(f"CTAs: {', '.join(page['ctas'][:5])}")

        lines.append("")  # blank line between pages

    lines.append("Now analyze this business and return the JSON object.")
    return "\n".join(lines)