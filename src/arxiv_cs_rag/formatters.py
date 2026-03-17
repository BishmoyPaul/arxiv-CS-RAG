from __future__ import annotations

from typing import Iterable

from .retrieval import PaperRecord


SEARCH_RESULTS_HEADER = "# Search Results\n\n"


def format_paper_markdown(paper: PaperRecord) -> str:
    title_line = f"### {paper.updated_display} | [{paper.title}]({paper.paper_url}) | [PDF]({paper.pdf_url})\n"
    authors_line = f"*{paper.authors}*  \n\n"
    return f"{title_line}{authors_line}{paper.abstract}\n\n---------------\n\n"


def format_prompt_entry(paper: PaperRecord) -> str:
    return f"<b> {paper.title} </b> \n Abstract: {paper.abstract}"


def format_search_results_markdown(papers: Iterable[PaperRecord]) -> str:
    return SEARCH_RESULTS_HEADER + "".join(format_paper_markdown(paper) for paper in papers)
