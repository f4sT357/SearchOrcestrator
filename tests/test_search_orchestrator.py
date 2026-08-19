from search_orchestrator import SearchResult, add_search_results, deduplicate_results


def test_add_search_results_ignores_duplicate_urls() -> None:
    results = [SearchResult(query="old", title="Old", url="https://example.com", snippet="")]
    added = add_search_results(results, [
        {"href": "https://example.com", "title": "Duplicate", "body": ""},
        {"href": "https://other.example", "title": "New", "body": "text"},
    ], "new query")
    assert added == 1
    assert [result.url for result in results] == ["https://example.com", "https://other.example"]


def test_deduplicate_results_removes_empty_and_duplicate_urls() -> None:
    results = [
        SearchResult(query="q", title="A", url="https://example.com", snippet=""),
        SearchResult(query="q", title="B", url="https://example.com", snippet=""),
        SearchResult(query="q", title="C", url="", snippet=""),
    ]
    assert [result.title for result in deduplicate_results(results)] == ["A"]
