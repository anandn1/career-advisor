from datetime import date

import pytest
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.agents import job_market_analyst as jma
from app.services import job_scraper

# --- Location matching ---


@pytest.mark.parametrize(
    ("wanted", "found", "expected"),
    [
        ("Pune", "Pune, Maharashtra, India", True),
        ("india", "Pune, Maharashtra, India", True),
        ("New York", "New York, NY", True),
        ("Pune, Maharashtra, India", "Pune", True),
        ("REMOTE", "remote", True),
        ("india", "Indianapolis, Indiana", False),
        ("Bangalore", "Pune", False),
        ("Bangalore", "", False),
        ("", "Pune", False),
        ("New Delhi", "New York", False),
    ],
)
def test_locations_match_on_whole_words(wanted, found, expected):
    assert jma._locations_match(wanted, found) is expected


# --- Cached posting parsing ---


def posting_document(title="Backend Dev", location="Pune", skills="go, r, c#, python", **meta):
    content = f"Job Title: {title}\nCompany: Acme\nLocation: {location}\nSkills Required: {skills}"
    metadata = {"title": title, "company": "Acme", "location": location} | meta
    return Document(page_content=content, metadata=metadata)


def test_short_skills_survive_in_cached_postings():
    job = jma._document_to_job(posting_document(skills="Go, R, C#, AI, Python"))
    assert job["skills_required"] == ["go", "r", "c#", "ai", "python"]


def test_cached_posting_without_skills_has_none():
    assert jma._skills_from_content("job title: x\ncompany: y") == []


def test_cached_jobs_are_labeled_with_their_scrape_date():
    job = jma._document_to_job(posting_document(scraped_at="2026-10-01"))
    assert (job["source"], job["scraped_at"]) == ("cached", "2026-10-01")


@pytest.mark.parametrize("title", [None, "", "N/A"])
def test_missing_title_falls_back_to_the_content(title):
    document = posting_document(title="Data Engineer")
    document.metadata["title"] = title
    assert jma._title_from(document) == "Data Engineer"


def test_missing_title_everywhere_uses_the_default():
    document = Document(page_content="Company: Acme", metadata={"title": None})
    assert jma._title_from(document) == jma.DEFAULT_TITLE


def test_stored_postings_record_the_scrape_date():
    document = jma._posting_to_document({"title": "Dev", "skills required": ["go"]})
    assert document.metadata["scraped_at"] == date.today().isoformat()


def test_stored_postings_round_trip_their_skills():
    document = jma._posting_to_document({"title": "Dev", "skills required": ["go", "docker"]})
    assert jma._document_to_job(document)["skills_required"] == ["go", "docker"]


# --- Deduplication ---


def job(company="Acme", skills=("go",), source="live"):
    return {"company": company, "skills_required": list(skills), "source": source}


def test_duplicate_jobs_keep_the_first():
    jobs = [job(source="live"), job(skills=("GO",), source="cached")]
    assert jma._dedupe_jobs(jobs) == [jobs[0]]


def test_jobs_with_different_skills_are_kept():
    assert len(jma._dedupe_jobs([job(skills=("go",)), job(skills=("rust",))])) == 2


# --- Graph nodes with fakes ---


class FakeStore:
    def __init__(self, fail=False):
        self.fail = fail
        self.added = []

    def add_documents(self, documents):
        if self.fail:
            raise RuntimeError("disk full")
        self.added.extend(documents)


@pytest.fixture
def scraper(monkeypatch):
    """Two live postings; one description fetch fails."""
    summaries = [
        {
            "title": "Dev A",
            "company": "Acme",
            "location": "Pune",
            "link": "https://x/a",
        },
        {
            "title": "Dev B",
            "company": "Beta",
            "location": "Pune",
            "link": "https://x/b",
        },
        {"title": "No link", "company": "Gamma", "location": "Pune"},
    ]

    def fetch_description(link):
        if link.endswith("/b"):
            raise job_scraper.ScraperError("blocked")
        return {"link": link, "skills required": ["go", "docker"]}

    monkeypatch.setattr(jma.job_scraper, "fetch_jobs", lambda role, location: summaries)
    monkeypatch.setattr(jma.job_scraper, "fetch_job_description", fetch_description)


def run_scrape(monkeypatch, cached_jobs=(), store=None):
    store = store or FakeStore()
    monkeypatch.setattr(jma, "_market_store", lambda: store)
    state = {"role": "Dev", "location": "Pune", "job_data": list(cached_jobs)}
    return jma.scrape_fresh_postings(state)["job_data"], store


def test_live_postings_are_labeled_live(scraper, monkeypatch):
    jobs, _ = run_scrape(monkeypatch)
    assert [(j["title"], j["source"]) for j in jobs] == [("Dev A", "live")]


def test_failed_and_linkless_postings_are_skipped(scraper, monkeypatch):
    jobs, _ = run_scrape(monkeypatch)
    assert [j["company"] for j in jobs] == ["Acme"]


def test_live_postings_are_cached(scraper, monkeypatch):
    _, store = run_scrape(monkeypatch)
    assert len(store.added) == 1


def test_cache_failure_keeps_the_live_results(scraper, monkeypatch):
    jobs, _ = run_scrape(monkeypatch, store=FakeStore(fail=True))
    assert [j["title"] for j in jobs] == ["Dev A"]


def test_cached_duplicate_of_a_live_posting_is_dropped(scraper, monkeypatch):
    cached = {
        "title": "Dev A",
        "company": "Acme",
        "skills_required": ["docker", "go"],
        "source": "cached",
    }
    jobs, _ = run_scrape(monkeypatch, cached_jobs=[cached])
    assert [j["source"] for j in jobs] == ["live"]


def test_cached_postings_remain_when_scraping_finds_nothing(monkeypatch):
    monkeypatch.setattr(jma.job_scraper, "fetch_jobs", lambda role, location: [])
    cached = job(company="Old Co", source="cached")
    jobs, _ = run_scrape(monkeypatch, cached_jobs=[cached])
    assert jobs == [cached]


def test_cached_search_filters_by_location(monkeypatch):
    documents = [
        posting_document(location="Pune"),
        posting_document(location=""),
        posting_document(location="Indianapolis"),
    ]

    class Retriever:
        def invoke(self, query):
            return documents

    monkeypatch.setattr(jma, "_retriever", lambda: Retriever())
    jobs = jma.search_cached_postings({"role": "Dev", "location": "Pune"})["job_data"]
    assert [j["location"] for j in jobs] == ["Pune"]


# --- Gap analysis ---


def analyze(user_skills, *job_skills):
    jobs = [
        {"title": f"J{i}", "company": "C", "skills_required": list(s)}
        for i, s in enumerate(job_skills)
    ]
    state = {
        "role": "Dev",
        "location": "Pune",
        "user_skills": user_skills,
        "job_data": jobs,
    }
    return jma.analyze_skill_gaps(state)["skill_gap_data"]


def test_gaps_are_skills_the_user_lacks():
    assert analyze(["Python"], ["python", "Docker"])["skill_gaps"] == ["docker"]


def test_skill_names_are_normalized_before_comparing():
    assert analyze(["react-js", "Node.JS"], ["React JS", "node.js."])["skill_gaps"] == []


def test_average_gaps_counts_every_job():
    assert analyze([], ["go", "rust"], [])["average_gaps_per_job"] == 1.0


def test_no_jobs_gives_zero_average():
    assert analyze(["go"])["average_gaps_per_job"] == 0


def test_graph_runs_the_three_steps_in_order():
    nodes = list(jma.get_job_market_graph().get_graph().nodes)
    assert nodes == [
        "__start__",
        "search_cached_postings",
        "scrape_fresh_postings",
        "analyze_skill_gaps",
        "__end__",
    ]


# --- Scraper ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We write Go and Python.", ["go", "python"]),
        ("Ready to go the extra mile", []),
        (
            "Experience with Docker, Kubernetes and PostgreSQL",
            ["docker", "kubernetes", "postgresql"],
        ),
        ("We use Slack daily", ["slack"]),
        ("a little slack in the schedule", []),
        ("DOCKER and docker", ["docker"]),
        ("", []),
    ],
)
def test_skill_extraction(text, expected):
    assert job_scraper.extract_skills(text) == expected


@pytest.mark.parametrize("page", [0, -1])
def test_page_must_be_positive(page):
    with pytest.raises(ValueError):
        job_scraper.fetch_jobs("Dev", "Pune", page=page)


def test_page_offsets_the_search(monkeypatch):
    seen = []
    monkeypatch.setattr(
        job_scraper,
        "_get_page",
        lambda url, params=None: seen.append(params) or BeautifulSoup("", "html.parser"),
    )
    job_scraper.fetch_jobs("Dev", "Pune", page=3)
    assert seen[0]["start"] == 2 * job_scraper.RESULTS_PER_PAGE


def test_remote_search_uses_the_remote_filter():
    params = job_scraper._search_params("Dev", "Remote", 0)
    assert params == {
        "keywords": "Dev",
        "start": 0,
        "f_WT": job_scraper.REMOTE_WORK_FILTER,
    }


def test_each_comma_separated_location_is_searched(monkeypatch):
    seen = []
    monkeypatch.setattr(
        job_scraper,
        "_get_page",
        lambda url, params=None: seen.append(params) or BeautifulSoup("", "html.parser"),
    )
    job_scraper.fetch_jobs("Dev", "Pune, ,Remote")
    assert [p.get("location", "remote") for p in seen] == ["Pune", "remote"]


def test_no_location_searches_worldwide(monkeypatch):
    seen = []
    monkeypatch.setattr(
        job_scraper,
        "_get_page",
        lambda url, params=None: seen.append(params) or BeautifulSoup("", "html.parser"),
    )
    job_scraper.fetch_jobs("Dev", None)
    assert seen[0]["location"] == "Worldwide"


def test_failed_location_does_not_stop_the_others(monkeypatch):
    html = (
        '<ul><li><div class="base-search-card"><a href="https://x/1"></a>'
        '<h3 class="base-search-card__title">Dev</h3></div></li></ul>'
    )

    def get_page(url, params=None):
        if params.get("location") == "Pune":
            raise job_scraper.ScraperError("blocked")
        return BeautifulSoup(html, "html.parser")

    monkeypatch.setattr(job_scraper, "_get_page", get_page)
    jobs = job_scraper.fetch_jobs("Dev", "Pune, Mumbai")
    assert [(j["title"], j["location"]) for j in jobs] == [("Dev", "Mumbai")]


def test_cards_without_a_link_are_skipped():
    html = '<div class="base-search-card"><h3>No link</h3></div>'
    card = BeautifulSoup(html, "html.parser").div
    assert job_scraper._parse_job_card(card, "Pune") is None


def test_card_fields_are_extracted():
    html = (
        '<div class="base-search-card"><a href="https://x/1"></a>'
        '<h3 class="base-search-card__title"> Dev </h3>'
        '<div class="base-search-card__info"><h4><a> Acme </a></h4></div>'
        "<div><time>2 days ago</time></div></div>"
    )
    card = BeautifulSoup(html, "html.parser").div
    assert job_scraper._parse_job_card(card, "Pune") == {
        "title": "Dev",
        "company": "Acme",
        "link": "https://x/1",
        "location": "Pune",
        "list_date": "2 days ago",
    }


def test_description_without_criteria_uses_placeholders(monkeypatch):
    monkeypatch.setattr(
        job_scraper, "_get_page", lambda url, params=None: BeautifulSoup("<p>hi</p>", "html.parser")
    )
    details = job_scraper.fetch_job_description("https://x/1")
    assert (
        details["seniority level"] == job_scraper.NOT_AVAILABLE and details["skills required"] == []
    )


def test_graph_executes_the_three_steps_in_order(monkeypatch):
    order = []
    for name in ("search_cached_postings", "scrape_fresh_postings", "analyze_skill_gaps"):
        monkeypatch.setattr(jma, name, lambda state, name=name: order.append(name) or {})
    jma.get_job_market_graph.cache_clear()
    jma.run_job_market_analysis("Dev", "Pune", [])
    assert order == ["search_cached_postings", "scrape_fresh_postings", "analyze_skill_gaps"]


def test_skills_with_commas_survive_the_cache(monkeypatch):
    posting = {"title": "Data Eng", "skills required": ["extract, transform, load", "sql"]}
    document = jma._posting_to_document(posting)
    assert jma._document_to_job(document)["skills_required"] == ["extract, transform, load", "sql"]


def test_older_cached_postings_without_skill_metadata_still_parse():
    assert jma._document_to_job(posting_document(skills="go, sql"))["skills_required"] == [
        "go",
        "sql",
    ]


def test_same_company_and_skills_with_different_titles_are_both_kept():
    jobs = [job() | {"title": "Backend Dev"}, job() | {"title": "SRE"}]
    assert len(jma._dedupe_jobs(jobs)) == 2


def test_only_the_first_postings_are_scraped(monkeypatch):
    summaries = [{"title": f"J{i}", "company": f"C{i}", "link": f"https://x/{i}"} for i in range(9)]
    fetched = []
    monkeypatch.setattr(jma.job_scraper, "fetch_jobs", lambda role, location: summaries)
    monkeypatch.setattr(
        jma.job_scraper,
        "fetch_job_description",
        lambda link: fetched.append(link) or {"skills required": []},
    )
    run_scrape(monkeypatch)
    assert len(fetched) == jma.POSTINGS_TO_SCRAPE


def test_network_errors_become_scraper_errors(monkeypatch):
    def timeout(*args, **kwargs):
        raise job_scraper.requests.Timeout("slow")

    monkeypatch.setattr(job_scraper.requests, "get", timeout)
    with pytest.raises(job_scraper.ScraperError):
        job_scraper.fetch_job_description("https://x/1")


def test_unreachable_search_returns_no_jobs(monkeypatch):
    def refused(*args, **kwargs):
        raise job_scraper.requests.ConnectionError("refused")

    monkeypatch.setattr(job_scraper.requests, "get", refused)
    assert job_scraper.fetch_jobs("Dev", "Pune") == []


DESCRIPTION_PAGE = """
<ul class="description__job-criteria-list">
  <li><span>Mid-Senior level</span></li><li><span>Full-time</span></li>
  <li><span>Engineering</span></li><li><span>Software</span></li>
</ul>
<div class="description__text description__text--rich"><section><div>
  We need Python, Docker and Kubernetes.
</div></section></div>
"""


def test_description_page_criteria_and_skills_are_extracted(monkeypatch):
    monkeypatch.setattr(
        job_scraper,
        "_get_page",
        lambda url, params=None: BeautifulSoup(DESCRIPTION_PAGE, "html.parser"),
    )
    details = job_scraper.fetch_job_description("https://x/1")
    assert details == {
        "link": "https://x/1",
        "skills required": ["docker", "kubernetes", "python"],
        "seniority level": "Mid-Senior level",
        "employment type": "Full-time",
        "job function": "Engineering",
        "industries": "Software",
    }


@pytest.mark.parametrize(
    ("wanted", "found", "expected"),
    [
        ("Pune, Mumbai", "Mumbai, Maharashtra, India", True),
        ("Pune, Mumbai", "Delhi", False),
        (" , ", "Pune", False),
    ],
)
def test_any_of_several_wanted_locations_matches(wanted, found, expected):
    assert jma._locations_match(wanted, found) is expected


@pytest.mark.parametrize("stored", ["not json", '"go"', "42", '["go", 3]', "null"])
def test_corrupt_skill_metadata_falls_back_to_the_content(stored):
    document = posting_document(skills="docker, sql")
    document.metadata["skills"] = stored
    assert jma._document_to_job(document)["skills_required"] == ["docker", "sql"]


def test_cached_skills_are_lowercased():
    document = posting_document()
    document.metadata["skills"] = '["Docker", "SQL"]'
    assert jma._document_to_job(document)["skills_required"] == ["docker", "sql"]


def test_same_title_and_skills_at_different_companies_are_both_kept():
    jobs = [job(company="Acme") | {"title": "Dev"}, job(company="Beta") | {"title": "Dev"}]
    assert len(jma._dedupe_jobs(jobs)) == 2
