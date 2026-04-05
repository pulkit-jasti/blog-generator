from typing import TypedDict, List, Optional
    
class BlogAgentState(TypedDict):
    # --- Input ---
    url: str                          # the URL we start with

    # --- Crawler output ---
    raw_pages: List[dict]             # list of {url, title, body, links}
    crawl_error: Optional[str]        # if crawling failed, we store why

    # --- Agent outputs (filled in later phases) ---
    business_summary: Optional[dict]  # Phase 2: analyst result
    topic_candidates: Optional[List[dict]]  # Phase 3: strategist result
    approved_topics: Optional[List[dict]]   # Phase 4: validator result
    rejection_reasons: Optional[List[str]]  # Phase 4: why topics were rejected
    retry_count: int                        # Phase 4: how many retries so far
    validation_passed: Optional[bool] 
    blog_briefs: Optional[List[dict]]       # Phase 5: writer result
    final_briefs: Optional[List[dict]]      # Phase 6: reviewer result