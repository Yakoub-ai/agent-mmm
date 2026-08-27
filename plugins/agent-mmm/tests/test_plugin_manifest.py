"""Tests for the plugin and marketplace manifests.

These files gate installation: a manifest Claude Code rejects means the plugin
cannot be installed at all, however good the rest of the repository is. That
failure is invisible from the repo — nothing imports these files, no other test
touches them, and the `$schema` URL they carried for months was a 404, so no
editor validated them either. They have been broken three separate times.

The schemas are vendored under `tests/schemas/` so this runs offline and in CI.
Refresh them from:
  https://json.schemastore.org/claude-code-plugin-manifest.json
  https://json.schemastore.org/claude-code-marketplace.json
"""
import json
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).parent.parent
REPO_ROOT = PLUGIN_ROOT.parent.parent
PLUGIN_MANIFEST = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
MARKETPLACE_MANIFEST = REPO_ROOT / ".claude-plugin" / "marketplace.json"
SCHEMA_DIR = Path(__file__).parent / "schemas"

# Component directories Claude Code auto-discovers at the plugin root.
AUTO_DISCOVERED = ("skills", "agents", "commands")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def plugin_manifest() -> dict:
    return _load(PLUGIN_MANIFEST)


@pytest.fixture(scope="module")
def marketplace_manifest() -> dict:
    return _load(MARKETPLACE_MANIFEST)


# --------------------------------------------------------------------------- #
# Schema conformance
# --------------------------------------------------------------------------- #
def _validate(doc: dict, schema_name: str) -> list[str]:
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load(SCHEMA_DIR / schema_name)
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{list(e.path) or '<root>'}: {e.message}"
        for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    ]


def test_plugin_manifest_matches_official_schema(plugin_manifest):
    errors = _validate(plugin_manifest, "claude-code-plugin-manifest.json")
    assert not errors, "plugin.json rejected by the official schema:\n" + "\n".join(errors)


def test_marketplace_manifest_matches_official_schema(marketplace_manifest):
    errors = _validate(marketplace_manifest, "claude-code-marketplace.json")
    assert not errors, "marketplace.json rejected by the official schema:\n" + "\n".join(errors)


# --------------------------------------------------------------------------- #
# The specific mistakes that have actually been made
# --------------------------------------------------------------------------- #
def test_component_paths_never_escape_the_plugin_root():
    """`../skills/` resolves outside the plugin and fails the `^\\./` pattern.

    This was the original break: paths were "fixed" from `./` to `../` on the
    theory that they resolve relative to `.claude-plugin/`. They do not — they
    resolve relative to the plugin root.
    """
    raw = PLUGIN_MANIFEST.read_text()
    assert '"../' not in raw, (
        "A component path starts with '../'. Paths resolve relative to the plugin "
        "root (the directory containing .claude-plugin/), not to .claude-plugin/ itself."
    )


def test_agents_is_not_declared_as_a_directory(plugin_manifest):
    """The `agents` field takes a path to a .md FILE, not a directory.

    `"agents": "./agents"` looks symmetrical with `skills` and `commands` and is
    the one that still fails: the schema requires `.*\\.md$`. Agent directories
    are only ever picked up by auto-discovery.
    """
    agents = plugin_manifest.get("agents")
    if agents is None:
        return
    paths = [agents] if isinstance(agents, str) else agents
    bad = [p for p in paths if not str(p).endswith(".md")]
    assert not bad, (
        f"`agents` entries must be .md files, got {bad}. To include the agents/ "
        "directory, omit the field entirely and let auto-discovery find it."
    )


def test_declared_component_paths_exist(plugin_manifest):
    """A declared path that points at nothing silently contributes nothing."""
    missing = []
    for field in ("skills", "agents", "commands", "hooks", "outputStyles"):
        value = plugin_manifest.get(field)
        if value is None or isinstance(value, dict):
            continue
        for rel in [value] if isinstance(value, str) else value:
            if not (PLUGIN_ROOT / str(rel).lstrip("./")).exists():
                missing.append(f"{field}: {rel}")
    assert not missing, f"Declared paths that do not exist: {missing}"


@pytest.mark.parametrize("directory", AUTO_DISCOVERED)
def test_auto_discovered_directories_exist(directory):
    """Auto-discovery only works if the directories are where it looks."""
    path = PLUGIN_ROOT / directory
    assert path.is_dir(), f"{directory}/ must exist at the plugin root for auto-discovery"
    assert any(path.rglob("*.md")), f"{directory}/ contains no markdown files"


# --------------------------------------------------------------------------- #
# Cross-manifest consistency
# --------------------------------------------------------------------------- #
def test_marketplace_source_points_at_this_plugin(marketplace_manifest, plugin_manifest):
    entry = next(
        p for p in marketplace_manifest["plugins"] if p["name"] == plugin_manifest["name"]
    )
    source = REPO_ROOT / str(entry["source"]).lstrip("./")
    assert source.is_dir(), f"marketplace source {entry['source']} is not a directory"
    assert (source / ".claude-plugin" / "plugin.json").is_file(), (
        f"marketplace source {entry['source']} has no .claude-plugin/plugin.json"
    )
    assert source.resolve() == PLUGIN_ROOT.resolve()


def test_schema_urls_resolve_to_a_real_schema(plugin_manifest, marketplace_manifest):
    """A `$schema` that 404s means no editor ever validates the file.

    The marketplace manifest pointed at anthropic.com for months; that URL is a
    404, which is why three broken manifests shipped without a warning.
    """
    for name, doc in (("plugin.json", plugin_manifest),
                      ("marketplace.json", marketplace_manifest)):
        url = doc.get("$schema")
        assert url, f"{name} has no $schema, so editors will not validate it"
        assert url.startswith("https://json.schemastore.org/"), (
            f"{name} $schema is {url!r}; use the schemastore URL, which resolves"
        )
