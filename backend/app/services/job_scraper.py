"""Scrapes public LinkedIn job listings and extracts required skills with spaCy."""

import logging
from functools import cache

import requests
import spacy
from bs4 import BeautifulSoup
from spacy.language import Language
from spacy.matcher import PhraseMatcher

REQUEST_TIMEOUT_SECONDS = 10
RESULTS_PER_PAGE = 25
SEARCH_URL = "https://in.linkedin.com/jobs/search"
REMOTE_WORK_FILTER = "2"
NOT_AVAILABLE = "NA"

logger = logging.getLogger(__name__)

KNOWN_SKILLS = [
    # Programming languages
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "PHP", "Ruby",
    "Swift", "Kotlin", "Rust", "Go", "SQL", "R", "MATLAB", "Scala", "Perl",
    "Bash", "Shell scripting", "HTML", "CSS", "SASS", "SCSS",
    # Web development
    "React.js", "Angular", "Vue.js", "Next.js", "Node.js", "Express.js",
    "Django", "Flask", "Laravel", "Spring Boot", "Ruby on Rails", "ASP.NET",
    "GraphQL", "RESTful APIs", "WebSocket", "jQuery", "Bootstrap",
    "Tailwind CSS", "WordPress", "Magento", "Shopify",
    # Data science and machine learning
    "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras",
    "Matplotlib", "Seaborn", "Plotly", "D3.js", "Data Visualization",
    "Data Wrangling", "Data Analysis", "Machine Learning", "Deep Learning",
    "Neural Networks", "Natural Language Processing", "NLP", "Computer Vision",
    "Time Series Analysis", "Statistical Modeling", "Data Mining",
    "AI/ML Algorithms", "Reinforcement Learning", "Generative Adversarial Networks", "GANs",
    # Cloud and DevOps
    "Amazon Web Services", "AWS", "Microsoft Azure", "Google Cloud Platform",
    "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins", "CI/CD",
    "Ansible", "Puppet", "Chef", "Vagrant", "OpenShift", "NGINX",
    "Load Balancing", "Monitoring", "Prometheus", "Grafana", "Serverless Architecture",
    "Microservices", "CloudFormation", "ElasticSearch",
    # Databases
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "SQLite",
    "DynamoDB", "MariaDB", "Oracle Database", "Microsoft SQL Server",
    "Couchbase", "Neo4j", "Graph Databases", "Firebase",
    # Operating systems
    "Linux", "Windows", "macOS", "Unix", "Android", "iOS", "Red Hat",
    "CentOS", "Ubuntu", "FreeBSD",
    # Version control and collaboration
    "Git", "GitHub", "GitLab", "Bitbucket", "Mercurial", "Subversion",
    "SVN", "JIRA", "Trello", "Asana", "Slack", "Confluence",
    # Mobile development
    "Android SDK", "iOS SDK", "React Native", "Flutter", "Xamarin",
    "SwiftUI", "Kotlin Multiplatform", "Ionic",
    # Cybersecurity
    "Information Security", "Network Security", "Application Security",
    "Cloud Security", "Penetration Testing", "Vulnerability Assessment",
    "OWASP", "Encryption", "Firewalls", "VPN",
    "Security Information and Event Management", "SIEM",
    "Identity and Access Management", "IAM", "Intrusion Detection Systems", "IDS",
    "Data Loss Prevention", "DLP",
    # Soft skills
    "Problem-solving", "Communication", "Teamwork", "Leadership",
    "Time Management", "Critical Thinking", "Project Management",
    "Agile Methodologies", "Scrum", "Kanban", "Collaboration",
    # Business intelligence and analytics
    "Power BI", "Tableau", "Looker", "QlikView", "Google Data Studio",
    "SAP Analytics Cloud", "Apache Hadoop", "Apache Spark", "Hive",
    "Pig", "Presto", "Microsoft Excel", "Advanced Excel",
    "Business Intelligence", "BI", "ETL", "Extract, Transform, Load",
    # Networking
    "TCP/IP", "DNS", "DHCP", "LAN", "WAN", "Network Protocols",
    "Wireless Networks", "Routing and Switching", "Firewall Configuration",
    "Virtual LAN", "VLAN",
    # Other technical skills
    "API Development", "Microservices Architecture", "Performance Tuning",
    "High Availability", "Load Testing", "Scalability", "SQL Optimization",
    "Automated Testing", "Unit Testing", "Integration Testing",
    "User Interface", "UI Design", "User Experience", "UX Design",
    "Cross-platform Development", "Event-driven Architecture",
    "Blockchain", "Internet of Things", "IoT", "Web Scraping",
    "WebRTC", "Augmented Reality", "AR", "Virtual Reality", "VR",
    "Robotics", "Game Development", "3D Modeling", "Embedded Systems",
    "Autonomous Systems",
    # Design and multimedia
    "Adobe Photoshop", "Adobe Illustrator", "Adobe Premiere Pro",
    "Adobe After Effects", "Figma", "Sketch", "Adobe XD",
    "Blender", "Unity", "Unreal Engine", "Final Cut Pro",
    "3D Rendering", "Animation",
]  # fmt: skip

CRITERIA_FIELDS = ("seniority level", "employment type", "job function", "industries")


class ScraperError(Exception):
    """Raised when LinkedIn cannot be reached or returns an unusable page."""


@cache
def _nlp() -> Language:
    return spacy.load("en_core_web_sm")


@cache
def _skill_matcher() -> PhraseMatcher:
    matcher = PhraseMatcher(_nlp().vocab)
    matcher.add("SKILLS", [_nlp()(skill.lower()) for skill in KNOWN_SKILLS])
    return matcher


def extract_skills(text: str) -> list[str]:
    doc = _nlp()(text.lower())
    return sorted({doc[start:end].text for _, start, end in _skill_matcher()(doc)})


def _get_page(url: str, params: dict | None = None) -> BeautifulSoup:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as error:
        raise ScraperError(f"Could not fetch {url}: {error}") from error
    return BeautifulSoup(response.content, "html.parser")


def _text_or(element, default: str) -> str:
    return element.get_text(strip=True) if element else default


def fetch_job_description(job_url: str) -> dict:
    """Fetch one job posting and return its criteria and extracted skills."""
    soup = _get_page(job_url)
    criteria = soup.find("ul", class_="description__job-criteria-list")
    description = soup.select_one("div.description__text.description__text--rich > section > div")
    description_text = description.get_text(separator=" ", strip=True) if description else ""

    details = {"link": job_url, "skills required": extract_skills(description_text)}
    for position, field in enumerate(CRITERIA_FIELDS, start=1):
        value = criteria.select_one(f"li:nth-child({position}) > span") if criteria else None
        details[field] = _text_or(value, NOT_AVAILABLE)
    return details


def _search_params(role: str, location: str, start: int) -> dict:
    params = {"keywords": role, "start": start}
    if location.lower() == "remote":
        params["f_WT"] = REMOTE_WORK_FILTER
    else:
        params["location"] = location
    return params


def _parse_job_card(card, location: str) -> dict | None:
    """Summary of one search result card, or None if it has no link to the posting."""
    anchor = card.select_one("a")
    if anchor is None or not anchor.get("href"):
        return None
    return {
        "title": _text_or(card.find("h3", class_="base-search-card__title"), "Unknown"),
        "company": _text_or(card.select_one("div.base-search-card__info > h4 > a"), "Unknown"),
        "link": anchor["href"],
        "location": location,
        "list_date": _text_or(card.select_one("div > time"), "Unknown"),
    }


def fetch_jobs(role: str, locations: str | None, page: int = 1) -> list[dict]:
    """Search listings for `role` in each comma-separated location (default: worldwide)."""
    location_list = [loc.strip() for loc in (locations or "Worldwide").split(",") if loc.strip()]
    start = (page - 1) * RESULTS_PER_PAGE
    jobs = []
    for location in location_list:
        try:
            soup = _get_page(SEARCH_URL, params=_search_params(role, location, start))
        except ScraperError:
            logger.warning("Job search failed for %s in %s", role, location)
            continue
        cards = soup.select("ul > li > div.base-search-card")
        jobs.extend(job for card in cards if (job := _parse_job_card(card, location)))
    return jobs
