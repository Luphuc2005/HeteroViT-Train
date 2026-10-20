import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from results_layout import NAV, find_result, logical_parts, resolve_result_path, result_roots


class ResultsLayoutTests(unittest.TestCase):
    def test_discovery_ignores_navigation_shortcuts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            flat = repo / 'results/flat_run'
            flat.mkdir(parents=True)
            grouped = repo / 'results' / NAV / '01_GROUP' / 'grouped_run'
            grouped.mkdir(parents=True)
            (grouped.parent / 'shortcut').symlink_to(flat, target_is_directory=True)
            self.assertEqual({p.name for p in result_roots(repo)}, {'flat_run', 'grouped_run'})
            self.assertEqual(find_result(repo, 'grouped_run'), grouped)

    def test_legacy_path_resolves_nested_file_after_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            target = repo / 'results' / NAV / '02_GROUP' / 'old_run' / 'ranks/rank0/train.csv'
            target.parent.mkdir(parents=True)
            target.write_text('epoch\n1\n')
            legacy = repo / 'results/old_run/ranks/rank0/train.csv'
            self.assertEqual(resolve_result_path(legacy, repo), target)
            self.assertEqual(resolve_result_path(target, repo), target)
            self.assertEqual(logical_parts(target.relative_to(repo / 'results')),
                             ('old_run', 'ranks', 'rank0', 'train.csv'))

    def test_duplicate_run_names_are_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / 'results/same').mkdir(parents=True)
            (repo / 'results' / NAV / '01_GROUP/same').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'found 2'):
                find_result(repo, 'same')


if __name__ == '__main__':
    unittest.main()
