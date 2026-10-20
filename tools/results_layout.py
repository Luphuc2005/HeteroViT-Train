"""Locate experiment bundles in either the historical flat or grouped layout."""
from pathlib import Path

NAV = '00_BY_EXPERIMENT'


def result_roots(repo):
    root = Path(repo) / 'results'
    loose = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink() and p.name != NAV]
    grouped = []
    if (root / NAV).is_dir():
        for group in (root / NAV).iterdir():
            if group.is_dir() and not group.is_symlink():
                grouped.extend(p for p in group.iterdir() if p.is_dir() and not p.is_symlink())
    return sorted(loose + grouped)


def find_result(repo, name):
    matches = [p for p in result_roots(repo) if p.name == name]
    if len(matches) != 1:
        raise ValueError('Expected one result %s, found %d' % (name, len(matches)))
    return matches[0]


def result_group(repo, group):
    path = Path(repo) / 'results' / NAV / group
    path.mkdir(parents=True, exist_ok=True)
    return path


def logical_parts(path):
    """Remove only the navigation/group prefix for historical classification."""
    parts = Path(path).parts
    return parts[2:] if parts and parts[0] == NAV else parts


def resolve_result_path(path, repo=None):
    path = Path(path)
    if path.exists():
        return path
    repo = Path(repo) if repo else Path(__file__).resolve().parents[1]
    parts = path.parts
    if 'results' not in parts:
        return path
    index = parts.index('results')
    tail = parts[index + 1:]
    if not tail or tail[0] == NAV:
        return path
    return find_result(repo, tail[0]).joinpath(*tail[1:])
