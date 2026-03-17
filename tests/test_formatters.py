from arxiv_cs_rag.formatters import format_paper_markdown, format_prompt_entry, format_search_results_markdown
from arxiv_cs_rag.retrieval import PaperRecord


def test_format_paper_markdown_contains_expected_fields():
    paper = PaperRecord(
        title="Test Paper",
        abstract="A useful abstract.",
        authors="Ada Lovelace, Alan Turing",
        updated_display="10 Mar 2026",
        paper_url="https://arxiv.org/abs/1234.5678",
        pdf_url="https://arxiv.org/pdf/1234.5678",
        source="Semantic Search",
        document_id="1234.5678",
    )

    markdown = format_paper_markdown(paper)

    assert "Test Paper" in markdown
    assert "Ada Lovelace, Alan Turing" in markdown
    assert "[PDF](https://arxiv.org/pdf/1234.5678)" in markdown


def test_format_prompt_entry_matches_current_shape():
    paper = PaperRecord(
        title="Prompt Paper",
        abstract="Important context",
        authors="Author One",
        updated_display="10 Mar 2026",
        paper_url="https://arxiv.org/abs/9999.9999",
        pdf_url="https://arxiv.org/pdf/9999.9999",
        source="Semantic Search",
    )

    prompt_entry = format_prompt_entry(paper)

    assert prompt_entry == "<b> Prompt Paper </b> \n Abstract: Important context"


def test_format_search_results_markdown_has_header():
    paper = PaperRecord(
        title="Header Paper",
        abstract="Abstract text",
        authors="Author One",
        updated_display="10 Mar 2026",
        paper_url="https://arxiv.org/abs/1111.1111",
        pdf_url="https://arxiv.org/pdf/1111.1111",
        source="Semantic Search",
    )

    markdown = format_search_results_markdown([paper])

    assert markdown.startswith("# Search Results")
