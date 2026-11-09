"""
Unit tests for cut_utils.read_branches() and cut_utils.iterate_branches() on a glob of ROOT
files.

Two small files with jagged tag_* / prt_* branches are written with uproot into tmp_path: a.root
holds events 0-1 and b.root events 2-3.
"""

import pytest
import uproot

import cut_utils
from conftest import make_tree

BRANCHES = ['tag_dtf_m', 'prt_idx_mom']


def write_tree(path, tree):
    """Write every field of `tree` as a branch of the TTree 'tree' in `path`."""
    with uproot.recreate(path) as f:
        f['tree'] = {field: tree[field] for field in tree.fields}


@pytest.fixture
def two_files(tmp_path):
    """a.root and b.root in tmp_path, b.root written first."""
    write_tree(tmp_path / 'b.root', make_tree(tag_dtf_m=[[30.0], [40.0, 41.0]],
                                              prt_idx_mom=[[0, 0], [0, 0, 1, 1]]))
    write_tree(tmp_path / 'a.root', make_tree(tag_dtf_m=[[10.0, 11.0], [20.0]],
                                              prt_idx_mom=[[0, 0, 1, 1], [0, 0]]))
    return tmp_path


def test_glob_reads_every_file_in_sorted_order(two_files):
    tree = cut_utils.read_branches(str(two_files / '*.root'), BRANCHES)
    assert tree['tag_dtf_m'].tolist() == [[10.0, 11.0], [20.0], [30.0], [40.0, 41.0]]
    assert tree['prt_idx_mom'].tolist() == [[0, 0, 1, 1], [0, 0], [0, 0], [0, 0, 1, 1]]


def test_entry_stop_crosses_the_file_boundary(two_files):
    tree = cut_utils.read_branches(str(two_files / '*.root'), BRANCHES, entry_stop=3)
    assert tree['tag_dtf_m'].tolist() == [[10.0, 11.0], [20.0], [30.0]]


def test_glob_matching_nothing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cut_utils.read_branches(str(tmp_path / '*.root'), BRANCHES)


def test_iterate_reads_every_file_in_sorted_order_one_chunk_at_a_time(two_files):
    chunks = list(cut_utils.iterate_branches(str(two_files / '*.root'), BRANCHES, step_size=1))
    assert [chunk['tag_dtf_m'].tolist() for chunk in chunks] == [
        [[10.0, 11.0]], [[20.0]], [[30.0]], [[40.0, 41.0]]]


def test_iterate_entry_stop_cuts_the_last_chunk(two_files):
    chunks = list(cut_utils.iterate_branches(str(two_files / '*.root'), BRANCHES, step_size=2,
                                             entry_stop=3))
    assert [chunk['tag_dtf_m'].tolist() for chunk in chunks] == [
        [[10.0, 11.0], [20.0]], [[30.0]]]


def test_iterate_glob_matching_nothing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(cut_utils.iterate_branches(str(tmp_path / '*.root'), BRANCHES, step_size=1))
