import pytest

from app.agents import skill_names


@pytest.mark.parametrize(
    ("gap", "user_skills"),
    [
        ("aws", ["AWS Lambda"]),
        ("docker", ["docker"]),
        ("React", ["react"]),
        ("ci/cd", ["CI/CD pipelines"]),
        ("node.js", ["Node.JS"]),
        ("microservices architecture", ["Microservices"]),
        ("Python programming", ["python"]),
        ("react", ["React Native"]),  # built on React
    ],
)
def test_a_skill_the_user_names_the_same_or_more_specifically_is_covered(gap, user_skills):
    assert skill_names.has_skill(gap, user_skills)


@pytest.mark.parametrize(
    ("gap", "user_skills"),
    [
        ("aws lambda", ["AWS"]),  # the user's skill is broader, so the gap stays
        ("java", ["JavaScript"]),
        ("c", ["C++"]),
        ("c", ["C#"]),
        (
            "web development",
            ["Web Scraping"],
        ),  # a qualifier word does not make it match
        ("sql", ["PostgreSQL"]),
        ("go", []),
        ("c", ["Objective-C"]),  # the gap must start the user's skill
        ("go", ["Go-to-market strategy"]),  # ...as a separate word
        ("r", ["R&D"]),
        ("software architecture", ["Software Development"]),  # different qualifiers
        ("web development", ["Web Frameworks"]),
    ],
)
def test_a_different_or_broader_skill_does_not_cover_the_gap(gap, user_skills):
    assert not skill_names.has_skill(gap, user_skills)


@pytest.mark.parametrize(
    "skill",
    [
        "Communication",
        "problem-solving",
        "Problem Solving",
        "communication skills",
        "Scrum",
    ],
)
def test_soft_skills_are_recognised_in_any_spelling(skill):
    assert skill_names.is_soft_skill(skill)


@pytest.mark.parametrize("skill", ["Kubernetes", "system design", "user research", "agile coach"])
def test_technical_skills_are_not_soft_skills(skill):
    assert not skill_names.is_soft_skill(skill)


def test_spellings_of_one_skill_join_under_its_shortest_name():
    groups = skill_names.group_skills(["Microservices Architecture", "microservices", "Kafka"])
    assert groups == {
        "kafka": "kafka",
        "microservices": "microservices",
        "microservices architecture": "microservices",
    }


def test_a_group_is_named_by_its_shortest_spelling_not_the_first_a_to_z():
    groups = skill_names.group_skills(["programming python", "python"])
    assert set(groups.values()) == {"python"}


def test_skills_that_share_a_word_stay_apart():
    groups = skill_names.group_skills(
        ["React", "React Native", "Spring", "Spring Boot", "Software Architecture",
         "Software Development"]
    )  # fmt: skip
    assert sorted(set(groups.values())) == [
        "react", "react native", "software architecture", "software development", "spring",
        "spring boot",
    ]  # fmt: skip


@pytest.mark.parametrize("blank", ["", "   ", "-", "\n", ".,;"])
def test_blank_or_punctuation_only_skills_are_left_out(blank):
    assert skill_names.group_skills([blank, "go"]) == {"go": "go"}
    assert not skill_names.has_skill(blank, ["anything"])
    assert not skill_names.is_soft_skill(blank)


@pytest.mark.parametrize("padded", [" docker\n", "\tDocker.", "docker;"])
def test_padding_and_trailing_punctuation_are_trimmed(padded):
    assert skill_names.normalize_skill(padded) == "docker"


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("Microservices Architecture", "microservices", True),
        ("python programming", "Python", True),
        ("Node.JS", "node.js", True),
        ("software architecture", "software development", False),
        ("web development", "web frameworks", False),
        ("react", "react native", False),
        ("", "", False),
    ],
)
def test_same_skill(a, b, same):
    assert skill_names.same_skill(a, b) is same
    assert skill_names.same_skill(b, a) is same
