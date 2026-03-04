"""
_academic.py
Fetch academic paper abstracts from OpenAlex for use as catademic input.

Usage via classify():
    classify(
        input_data=None,          # omit when using journal_issn
        categories=[...],
        journal_issn="0894-4393", # ISSN of the journal to pull from
        paper_limit=50,           # number of papers to fetch
        date_from="2020-01-01",   # optional start date (YYYY-MM-DD)
        date_to="2024-12-31",     # optional end date (YYYY-MM-DD)
    )

Supported sources: "openalex"
"""

import urllib.parse

import requests
import pandas as pd

_OPENALEX_SOURCES = "https://api.openalex.org/sources"
_OPENALEX_CONCEPTS = "https://api.openalex.org/concepts"

SUPPORTED_SOURCES = ["openalex"]

_OPENALEX_BASE = "https://api.openalex.org/works"
_PAGE_SIZE = 200  # OpenAlex max per page

# All top-level fields we want; nested structs are unpacked in _parse_work()
_SELECT_FIELDS = (
    "id,doi,title,publication_year,publication_date,language,type,"
    "abstract_inverted_index,cited_by_count,fwci,citation_normalized_percentile,"
    "is_retracted,referenced_works_count,"
    "primary_location,biblio,open_access,authorships,primary_topic,keywords,concepts"
)


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions = [
        (pos, word)
        for word, positions in inverted_index.items()
        for pos in positions
    ]
    return " ".join(w for _, w in sorted(word_positions))


def _parse_work(work: dict) -> dict:
    """Flatten a single OpenAlex work object into a row dict."""
    # --- core ---
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    # --- citation stats ---
    cnp = work.get("citation_normalized_percentile") or {}

    # --- bibliographic ---
    biblio = work.get("biblio") or {}

    # --- open access ---
    oa = work.get("open_access") or {}

    # --- journal / source ---
    primary_loc = work.get("primary_location") or {}
    source = primary_loc.get("source") or {}

    # --- authors, institutions, countries ---
    authorships = work.get("authorships") or []
    authors = "; ".join(
        a["author"]["display_name"]
        for a in authorships
        if a.get("author", {}).get("display_name")
    )
    institutions = "; ".join(
        inst["display_name"]
        for a in authorships
        for inst in (a.get("institutions") or [])
        if inst.get("display_name")
    )
    countries = "; ".join(sorted({
        c
        for a in authorships
        for c in (a.get("countries") or [])
    }))

    # --- topic hierarchy ---
    pt = work.get("primary_topic") or {}
    topic = pt.get("display_name", "")
    topic_subfield = (pt.get("subfield") or {}).get("display_name", "")
    topic_field = (pt.get("field") or {}).get("display_name", "")
    topic_domain = (pt.get("domain") or {}).get("display_name", "")

    # --- keywords & concepts ---
    keywords = "; ".join(k["display_name"] for k in (work.get("keywords") or []))
    concepts = "; ".join(c["display_name"] for c in (work.get("concepts") or []))

    return {
        # --- text (primary input to LLM) ---
        "text":                     abstract,
        # --- identifiers ---
        "openalex_id":              work.get("id", ""),
        "doi":                      work.get("doi", ""),
        # --- core metadata ---
        "title":                    work.get("title", ""),
        "publication_date":         work.get("publication_date", ""),
        "publication_year":         work.get("publication_year"),
        "type":                     work.get("type", ""),
        "language":                 work.get("language", ""),
        # --- journal ---
        "journal_name":             source.get("display_name", ""),
        "is_oa":                    oa.get("is_oa", False),
        "oa_status":                oa.get("oa_status", ""),
        # --- bibliographic ---
        "volume":                   biblio.get("volume"),
        "issue":                    biblio.get("issue"),
        "first_page":               biblio.get("first_page"),
        "last_page":                biblio.get("last_page"),
        # --- citation metrics ---
        "cited_by_count":           work.get("cited_by_count", 0),
        "fwci":                     work.get("fwci"),
        "citation_percentile":      cnp.get("value"),
        "is_in_top_10_percent":     cnp.get("is_in_top_10_percent", False),
        "is_in_top_1_percent":      cnp.get("is_in_top_1_percent", False),
        "referenced_works_count":   work.get("referenced_works_count", 0),
        # --- quality flags ---
        "is_retracted":             work.get("is_retracted", False),
        # --- authorship ---
        "authors":                  authors,
        "institutions":             institutions,
        "author_countries":         countries,
        # --- topic hierarchy ---
        "topic":                    topic,
        "topic_subfield":           topic_subfield,
        "topic_field":              topic_field,
        "topic_domain":             topic_domain,
        # --- classification tags ---
        "keywords":                 keywords,
        "concepts":                 concepts,
    }


def fetch_academic_papers(
    journal_issn: str = None,
    limit: int = 50,
    date_from: str = None,
    date_to: str = None,
    polite_email: str = None,
    journal_name: str = None,
    journal_field: str = None,
    topic_name: str = None,
    topic_id: str = None,
) -> pd.DataFrame:
    """
    Fetch papers from OpenAlex by journal and/or topic.

    Filters can be combined — e.g. demography-tagged papers specifically in Socius.
    Paginates automatically using OpenAlex cursor pagination so limits
    above 200 work correctly.

    Args:
        journal_issn:  The journal ISSN (e.g. "0894-4393"). Use this or journal_name.
        journal_name:  Journal name to search for (e.g. "socius"). Resolves to an
                       ISSN automatically via find_journal(). If multiple journals
                       match, the top result is used — check the printed output to
                       confirm the right journal was selected, or use journal_issn
                       directly for an exact match.
        journal_field: Discipline name to pull papers from all matching journals
                       (e.g. "demography", "sociology"). Fetches across every journal
                       whose name matches the field. Use find_journals_by_field() to
                       preview which journals will be included.
        topic_name:    Filter papers by content topic (e.g. "demography",
                       "machine learning"). Filters individual article content, not
                       the journal type. Resolves via find_topic(). Can be combined
                       with any journal filter.
        topic_id:      OpenAlex concept ID (e.g. "https://openalex.org/C149923435").
                       Use this or topic_name for an exact match.
        limit:         Maximum number of papers to return. Default 50.
        date_from:     Optional start date filter as "YYYY-MM-DD".
        date_to:       Optional end date filter as "YYYY-MM-DD".
        polite_email:  Optional email address for OpenAlex's polite pool
                       (faster rate limits). E.g. "you@example.com".

    Returns:
        DataFrame with columns:
            text, openalex_id, doi, title, publication_date, publication_year,
            type, language, journal_name, is_oa, oa_status, volume, issue,
            first_page, last_page, cited_by_count, fwci, citation_percentile,
            is_in_top_10_percent, is_in_top_1_percent, referenced_works_count,
            is_retracted, authors, institutions, author_countries,
            topic, topic_subfield, topic_field, topic_domain,
            keywords, concepts
    """
    n_journal_params = sum(x is not None for x in [journal_issn, journal_name, journal_field])
    if n_journal_params > 1:
        raise ValueError(
            "[CatAdemic] Pass only one of journal_issn, journal_name, or journal_field."
        )
    if journal_name is not None:
        matches = find_journal(journal_name)
        top = matches.iloc[0]
        journal_issn = top["issn"]
        print(f"[CatAdemic] Resolved '{journal_name}' → {top['display_name']} (ISSN: {journal_issn})")
        if len(matches) > 1:
            other_names = ", ".join(matches["display_name"].iloc[1:4].tolist())
            print(f"  Other matches: {other_names}")
            print(f"  Use journal_issn= directly if this isn't the right journal.\n")

    journal_issns = None  # list of ISSNs for journal_field OR filter
    if journal_field is not None:
        field_journals = find_journals_by_field(journal_field)
        journal_issns = field_journals["issn"].dropna().tolist()
        names = ", ".join(field_journals["display_name"].iloc[:4].tolist())
        print(f"[CatAdemic] Found {len(journal_issns)} journals for field '{journal_field}': {names}{'...' if len(journal_issns) > 4 else ''}\n")

    if topic_name is not None and topic_id is not None:
        raise ValueError(
            "[CatAdemic] Pass either topic_name or topic_id, not both."
        )
    if topic_name is not None:
        matches = find_topic(topic_name)
        top = matches.iloc[0]
        topic_id = top["id"]
        print(f"[CatAdemic] Resolved '{topic_name}' → {top['display_name']} ({top['works_count']:,} works)")
        if len(matches) > 1:
            other_names = ", ".join(matches["display_name"].iloc[1:4].tolist())
            print(f"  Other matches: {other_names}")
            print(f"  Use topic_id= directly if this isn't the right field.\n")

    if journal_issn is None and journal_issns is None and topic_id is None:
        raise ValueError(
            "[CatAdemic] Provide at least one of: journal_issn, journal_name, "
            "journal_field, topic_name, or topic_id."
        )

    filters = []
    if journal_issn:
        filters.append(f"primary_location.source.issn:{journal_issn}")
    elif journal_issns:
        filters.append(f"primary_location.source.issn:{'|'.join(journal_issns)}")
    if topic_id:
        filters.append(f"concepts.id:{topic_id}")
    if date_from and date_to:
        filters.append(f"from_publication_date:{date_from},to_publication_date:{date_to}")
    elif date_from:
        filters.append(f"from_publication_date:{date_from}")
    elif date_to:
        filters.append(f"to_publication_date:{date_to}")

    headers = {}
    if polite_email:
        headers["User-Agent"] = f"catademic/1.0 (mailto:{polite_email})"

    params = {
        "filter": ",".join(filters),
        "sort": "publication_date:desc",
        "per-page": min(limit, _PAGE_SIZE),
        "select": _SELECT_FIELDS,
        "cursor": "*",  # enable cursor pagination
    }

    rows = []
    first_page = True

    while len(rows) < limit:
        r = requests.get(_OPENALEX_BASE, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        body = r.json()

        # On first page: print summary and validate results exist
        if first_page:
            first_page = False
            meta = body.get("meta", {})
            total_available = meta.get("count", 0)

            if total_available == 0:
                raise ValueError(
                    f"[CatAdemic] No articles found for ISSN '{journal_issn}'. "
                    "Check the ISSN with cat.find_journal() and try again."
                )

            # Get journal name from first result
            first_result = body.get("results", [{}])[0]
            journal_name = (
                first_result.get("primary_location", {})
                .get("source", {})
                .get("display_name", journal_issn)
            )

            # Get oldest publication date for date range info (requires a separate
            # asc-sorted query — use the last item on the current page as a proxy)
            results_on_page = body.get("results", [])
            newest = results_on_page[0].get("publication_date", "unknown") if results_on_page else "unknown"
            oldest_on_page = results_on_page[-1].get("publication_date", "unknown") if results_on_page else "unknown"

            date_range_str = ""
            if date_from or date_to:
                parts = []
                if date_from:
                    parts.append(f"from {date_from}")
                if date_to:
                    parts.append(f"to {date_to}")
                date_range_str = f"  Date filter:        {' '.join(parts)}\n"

            will_fetch = min(limit, total_available)
            print(
                f"\n[CatAdemic] Journal:            {journal_name}\n"
                f"  ISSN:               {journal_issn}\n"
                f"  Total available:    {total_available:,} articles\n"
                f"  Most recent:        {newest}\n"
                f"{date_range_str}"
                f"  Fetching:           {will_fetch} articles (sorted newest first)\n"
            )

        for work in body.get("results", []):
            parsed = _parse_work(work)
            if parsed["text"]:  # skip papers with no abstract
                rows.append(parsed)
            if len(rows) >= limit:
                break

        next_cursor = body.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break

        params = {
            "filter": ",".join(filters),
            "sort": "publication_date:desc",
            "per-page": min(limit - len(rows), _PAGE_SIZE),
            "select": _SELECT_FIELDS,
            "cursor": next_cursor,
        }

    return pd.DataFrame(rows)


# TODO (journal_field): The current implementation searches journal names/text, which
# misses journals that cover a field without naming it (e.g. "Population Studies" for
# demography). The proper fix is to ship a bundled journal dataset as a lookup table:
#   1. Run social_science_computer_review/pull_all_journals.py to fetch ~17k journals
#      (filter: impact_factor > 1) with topics/field classification from OpenAlex.
#   2. LLM-classify all 250k journals by field using cat-ademic itself, and publish
#      the result as a standalone HuggingFace dataset for the research community.
#   3. Bundle the filtered CSV (or a field→issn mapping derived from it) into the
#      cat-ademic package as a static asset.
#   4. Replace the name-search below with a local lookup against that asset.
# See: JOURNAL_DATASET_PLAN.md for full task outline.
def find_journals_by_field(field: str, limit: int = 20) -> pd.DataFrame:
    """
    Search OpenAlex for journals whose name or focus matches a field (e.g. "demography").

    This finds journals *classified as* belonging to a field — useful for fetching
    papers across all journals in a discipline, rather than filtering article content
    (which is what topic_name does).

    Note: Current implementation uses name-based search and may miss journals that
    cover a field without naming it. See JOURNAL_DATASET_PLAN.md for the planned
    improvement using a bundled empirical field → journal mapping.

    Args:
        field: Field name to search for (e.g. "demography", "sociology", "economics").
        limit: Maximum number of journals to return. Default 20.

    Returns:
        DataFrame with columns: display_name, issn, works_count, publisher

    Example:
        >>> cat.find_journals_by_field("demography")
        # Then fetch across all demography journals:
        >>> cat.fetch_academic_papers(journal_field="demography", limit=100)
    """
    print(f"[CatAdemic] Searching for journals in field: '{field}'...")
    r = requests.get(
        _OPENALEX_SOURCES,
        params={
            "search": field,
            "filter": "type:journal",
            "per-page": limit,
            "select": "display_name,issn_l,issn,works_count,host_organization_name",
        },
        timeout=30,
    )
    r.raise_for_status()
    results = r.json().get("results", [])

    if not results:
        raise ValueError(
            f"[CatAdemic] No journals found for field '{field}'. "
            "Try a different search term."
        )

    rows = [{
        "display_name": s["display_name"],
        "issn":         s.get("issn_l") or (s.get("issn") or [""])[0],
        "works_count":  s.get("works_count", 0),
        "publisher":    s.get("host_organization_name", ""),
    } for s in results if s.get("issn_l") or s.get("issn")]

    return pd.DataFrame(rows, columns=["display_name", "issn", "works_count", "publisher"])


def find_topic(name: str, limit: int = 10) -> pd.DataFrame:
    """
    Search OpenAlex for academic fields/concepts matching a name string.

    Use this to discover the topic_id needed for fetch_academic_papers() and the
    topic_name parameter on classify(), extract(), and explore().

    Args:
        name:  Field or concept name to search for (e.g. "demography",
               "machine learning", "political polarization").
        limit: Maximum number of results to return. Default 10.

    Returns:
        DataFrame with columns:
            display_name, id, level, works_count
        Level guide: 0=domain, 1=field, 2=subfield, 3+=specific topic

    Example:
        >>> import catademic as cat
        >>> cat.find_topic("demography")
        # Pick the id from the results, then:
        >>> cat.classify(categories=[...], topic_name="demography", api_key="...")
    """
    print(f"[CatAdemic] Searching fields for: '{name}'...")
    r = requests.get(
        _OPENALEX_CONCEPTS,
        params={"search": name, "per-page": limit, "select": "display_name,id,level,works_count"},
        timeout=30,
    )
    r.raise_for_status()
    results = r.json().get("results", [])

    if not results:
        raise ValueError(
            f"[CatAdemic] No fields found matching '{name}'. "
            "Try a shorter or different search term."
        )

    rows = [{
        "display_name": s["display_name"],
        "id":           s["id"],
        "level":        s.get("level"),
        "works_count":  s.get("works_count", 0),
    } for s in results]

    return pd.DataFrame(rows, columns=["display_name", "id", "level", "works_count"])


def find_journal(name: str, limit: int = 10) -> pd.DataFrame:
    """
    Search OpenAlex for journals matching a name string.

    Use this to discover the ISSN needed for fetch_academic_papers() and the
    journal_issn parameter on classify(), extract(), and explore().

    Args:
        name:  Journal name or partial name to search for (e.g. "socius",
               "social science computer", "nature human behaviour").
        limit: Maximum number of results to return. Default 10.

    Returns:
        DataFrame with columns:
            display_name, issn, works_count, cited_by_count, is_oa, publisher

    Example:
        >>> import catademic as cat
        >>> cat.find_journal("socius")
        # Pick the ISSN from the results, then:
        >>> cat.classify(categories=[...], journal_issn="2378-0231", api_key="...")
    """
    params = {
        "search": name,
        "per-page": limit,
        "select": "display_name,issn_l,issn,works_count,cited_by_count,is_oa,host_organization_name",
    }
    print(f"[CatAdemic] Searching journals for: '{name}'...")
    r = requests.get(_OPENALEX_SOURCES, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])

    if not results:
        raise ValueError(
            f"[CatAdemic] No journals found matching '{name}'. "
            "Try a shorter or different search term."
        )

    rows = []
    for s in results:
        issns = s.get("issn") or []
        rows.append({
            "display_name":  s.get("display_name", ""),
            "issn":          s.get("issn_l", issns[0] if issns else ""),
            "all_issns":     "; ".join(issns),
            "works_count":   s.get("works_count", 0),
            "cited_by_count": s.get("cited_by_count", 0),
            "is_oa":         s.get("is_oa", False),
            "publisher":     s.get("host_organization_name", ""),
        })

    return pd.DataFrame(rows, columns=[
        "display_name", "issn", "all_issns", "works_count", "cited_by_count", "is_oa", "publisher"
    ])
