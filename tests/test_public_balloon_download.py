import pytest

from scripts.download_public_balloon import valid_project_slug


@pytest.mark.parametrize("project", ["red-baloons/blue_balloons", "workspace/project", "a/b", "workspace_1/project-v2"])
def test_project_slug_accepts_literal_public_identifiers(project):
    assert valid_project_slug(project)


@pytest.mark.parametrize("project", ["../project", "a/../b", "a/b/c", "a/b?api_key=secret",
    "https://api.roboflow.com/a/b", "a/b#fragment", "a/%2e%2e", "a\\b", "a/", "/b", "a/b\n", None])
def test_project_slug_rejects_paths_and_url_syntax(project):
    assert not valid_project_slug(project)
