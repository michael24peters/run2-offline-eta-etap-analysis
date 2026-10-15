"""
Unit tests for cut_utils.tos_mask().

TOS means the candidate is TOS on at least one line at every trigger level:
AND across levels, OR within a level. A line that was never evaluated in an
event is stored as an empty list and reads as "did not fire".
"""

import awkward as ak
import pytest

import cut_utils
from conftest import make_tree

LEVELS = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],
    'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos1'],
    'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_tos1'],
}


def tree_with(**overrides):
    """One event, one candidate, TOS on the first line of every level. Each
    test changes only the branches it is about."""
    fields = dict(tag_dtf_m=[[550.0]],
                  tag_l0_tos0=[[1.0]], tag_l0_tos1=[[0.0]],
                  tag_hlt1_tos0=[[1.0]], tag_hlt1_tos1=[[0.0]],
                  tag_hlt2_tos0=[[1.0]], tag_hlt2_tos1=[[0.0]])
    fields.update(overrides)
    return make_tree(**fields)


# --- AND across levels, OR within a level ------------------------------------

def test_tos_at_every_level_passes():
    mask, _ = cut_utils.tos_mask(tree_with(), LEVELS)
    assert mask.tolist() == [[True]]


@pytest.mark.parametrize('level_branches', [
    ('tag_l0_tos0', 'tag_l0_tos1'),
    ('tag_hlt1_tos0', 'tag_hlt1_tos1'),
    ('tag_hlt2_tos0', 'tag_hlt2_tos1'),
])
def test_missing_any_one_level_fails(level_branches):
    """TOS at two levels out of three is not TOS."""
    tree = tree_with(**{b: [[0.0]] for b in level_branches})
    mask, _ = cut_utils.tos_mask(tree, LEVELS)
    assert mask.tolist() == [[False]]


def test_either_line_satisfies_its_level():
    """Within a level the lines are OR'd: the second line alone is enough."""
    tree = tree_with(tag_l0_tos0=[[0.0]], tag_l0_tos1=[[1.0]],
                     tag_hlt2_tos0=[[0.0]], tag_hlt2_tos1=[[1.0]])
    mask, _ = cut_utils.tos_mask(tree, LEVELS)
    assert mask.tolist() == [[True]]


def test_lines_from_different_levels_do_not_substitute():
    """Two L0 lines firing do not make up for a missing HLT1 line."""
    tree = tree_with(tag_l0_tos1=[[1.0]], tag_hlt1_tos0=[[0.0]])
    mask, _ = cut_utils.tos_mask(tree, LEVELS)
    assert mask.tolist() == [[False]]


def test_per_candidate_within_an_event():
    """Each candidate is judged on its own entries."""
    tree = make_tree(tag_dtf_m=[[550.0, 551.0]],
                     tag_l0_tos0=[[1.0, 1.0]], tag_l0_tos1=[[0.0, 0.0]],
                     tag_hlt1_tos0=[[1.0, 0.0]], tag_hlt1_tos1=[[0.0, 1.0]],
                     tag_hlt2_tos0=[[1.0, 0.0]], tag_hlt2_tos1=[[0.0, 0.0]])
    mask, _ = cut_utils.tos_mask(tree, LEVELS)
    assert mask.tolist() == [[True, False]]


# --- absent lines --------------------------------------------------------------

def test_absent_line_reads_as_not_fired():
    """An empty list means the line was not evaluated; the other line of the
    level still decides it, and the mask keeps candidate shape."""
    tree = make_tree(tag_dtf_m=[[550.0, 551.0], [552.0]],
                     tag_l0_tos0=[[1.0, 1.0], [1.0]], tag_l0_tos1=[[0.0, 0.0], [0.0]],
                     tag_hlt1_tos0=[[1.0, 1.0], [1.0]], tag_hlt1_tos1=[[0.0, 0.0], [0.0]],
                     tag_hlt2_tos0=[[], [1.0]], tag_hlt2_tos1=[[1.0, 0.0], []])
    mask, report = cut_utils.tos_mask(tree, LEVELS)
    assert mask.tolist() == [[True, False], [True]]
    assert report['HLT2']['not_evaluated_in_events'] == {'tag_hlt2_tos0': 1,
                                                        'tag_hlt2_tos1': 1}
    assert report['HLT2']['ignored'] == []


def test_level_with_every_line_absent_passes_nothing():
    tree = tree_with(tag_hlt2_tos0=[[]], tag_hlt2_tos1=[[]])
    mask, report = cut_utils.tos_mask(tree, LEVELS)
    assert mask.tolist() == [[False]]
    assert report['HLT2']['ignored'] == ['tag_hlt2_tos0', 'tag_hlt2_tos1']


# --- report ------------------------------------------------------------------

def test_report_counts_per_level_and_cumulative():
    tree = make_tree(tag_dtf_m=[[550.0, 551.0, 552.0]],
                     tag_l0_tos0=[[1.0, 1.0, 1.0]], tag_l0_tos1=[[0.0, 0.0, 0.0]],
                     tag_hlt1_tos0=[[1.0, 1.0, 0.0]], tag_hlt1_tos1=[[0.0, 0.0, 0.0]],
                     tag_hlt2_tos0=[[1.0, 0.0, 1.0]], tag_hlt2_tos1=[[0.0, 0.0, 0.0]])
    mask, report = cut_utils.tos_mask(tree, LEVELS)
    assert mask.tolist() == [[True, False, False]]
    assert [report[lv]['candidates_tos'] for lv in LEVELS] == [3, 2, 2]
    assert [report[lv]['candidates_tos_cumulative'] for lv in LEVELS] == [3, 2, 1]
    assert list(report) == ['L0', 'HLT1', 'HLT2']


def test_single_level():
    """One level is just an OR over its lines."""
    tree = tree_with(tag_l0_tos0=[[0.0]])
    mask, _ = cut_utils.tos_mask(tree, {'HLT1': LEVELS['HLT1']})
    assert mask.tolist() == [[True]]


# --- input checks ------------------------------------------------------------

def test_flat_list_is_rejected():
    """A flat list of branches has no levels; it must not be OR'd silently."""
    with pytest.raises(TypeError):
        cut_utils.tos_mask(tree_with(), [b for v in LEVELS.values() for b in v])


def test_level_without_lines_is_rejected():
    with pytest.raises(ValueError):
        cut_utils.tos_mask(tree_with(), {**LEVELS, 'HLT2': []})


def test_empty_tree():
    tree = make_tree(**{b: [] for b in ['tag_dtf_m'] + sum(LEVELS.values(), [])})
    mask, report = cut_utils.tos_mask(tree, LEVELS)
    assert ak.num(mask, axis=0) == 0
    assert report['HLT2']['candidates_tos_cumulative'] == 0
