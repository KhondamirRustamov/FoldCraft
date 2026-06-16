"""Tests for the BindCraft locator (bindcraft_deps.py).

These exercise only the path-resolution logic, which is deliberately free of
PyRosetta/BindCraft imports, so they run on a plain CPU machine. We fake a
BindCraft checkout on disk rather than cloning the real one.
"""
import os
import sys

import pytest

import bindcraft_deps as bd


def _make_fake_bindcraft(root, with_dalphaball=True):
    """Create a minimal BindCraft checkout layout under ``root``/BindCraft."""
    repo = os.path.join(root, "BindCraft")
    functions = os.path.join(repo, "functions")
    os.makedirs(functions)
    open(os.path.join(functions, "__init__.py"), "w").close()
    if with_dalphaball:
        open(os.path.join(functions, "DAlphaBall.gcc"), "w").close()
    return repo


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    # Ensure a stray BINDCRAFT_PATH in the real environment can't leak in.
    monkeypatch.delenv("BINDCRAFT_PATH", raising=False)


class TestFindBindcraftRepo:
    def test_explicit_path_is_the_repo(self, tmp_path):
        repo = _make_fake_bindcraft(tmp_path)
        assert bd.find_bindcraft_repo(repo) == os.path.abspath(repo)

    def test_explicit_path_is_the_parent(self, tmp_path):
        repo = _make_fake_bindcraft(tmp_path)
        # Passing the parent dir should also resolve to the nested BindCraft.
        assert bd.find_bindcraft_repo(str(tmp_path)) == os.path.abspath(repo)

    def test_env_var_used(self, tmp_path, monkeypatch):
        repo = _make_fake_bindcraft(tmp_path)
        monkeypatch.setenv("BINDCRAFT_PATH", str(tmp_path))
        assert bd.find_bindcraft_repo() == os.path.abspath(repo)

    def test_missing_raises_with_instructions(self, tmp_path):
        empty = tmp_path / "nothing"
        empty.mkdir()
        with pytest.raises(ImportError) as exc:
            bd.find_bindcraft_repo(str(empty))
        msg = str(exc.value)
        assert "git clone" in msg and "BindCraft" in msg


class TestEnsureImportable:
    def test_inserts_parent_on_syspath_and_is_idempotent(self, tmp_path):
        repo = _make_fake_bindcraft(tmp_path)
        parent = os.path.dirname(repo)
        original = list(sys.path)
        try:
            bd.ensure_bindcraft_importable(repo)
            assert parent in sys.path
            count_after_first = sys.path.count(parent)
            bd.ensure_bindcraft_importable(repo)
            assert sys.path.count(parent) == count_after_first  # no duplicate
        finally:
            sys.path[:] = original


class TestDalphaballPath:
    def test_returns_bundled_binary(self, tmp_path):
        repo = _make_fake_bindcraft(tmp_path, with_dalphaball=True)
        expected = os.path.join(repo, "functions", "DAlphaBall.gcc")
        assert bd.dalphaball_path(repo) == expected

    def test_missing_binary_raises(self, tmp_path, monkeypatch):
        repo = _make_fake_bindcraft(tmp_path, with_dalphaball=False)
        # Run from a dir with no ./DAlphaBall.gcc fallback either.
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            bd.dalphaball_path(repo)
