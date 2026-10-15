"""
Unit tests for cut_utils.drop_malformed().

The gatekeeper: everything downstream assumes a well-formed tree, so this is
the one function whose job is to reject input. It drops an event for exactly
four reasons, and derives which branches to check from their name prefixes.

Each of the four reasons gets its own test with only that fault present, plus
tests that the reasons are attributed to the right counter -- a filter that
drops the right events for the wrong stated reason is a debugging trap.
"""

from conftest import make_tree

import cut_utils


def good_event():
    """One well-formed event: 2 candidates, 4 daughters, one trigger entry per
    candidate. Every test below starts from this and breaks one thing."""
    return dict(
        tag_dtf_m=[[550.0, 549.0]],
        tag_hlt2_tos0=[[1.0, 0.0]],
        prt_idx_mom=[[0, 0, 1, 1]],
        prt_pid=[[-13, 13, -13, 13]],
    )


def test_well_formed_event_is_kept():
    """Baseline: nothing wrong, nothing dropped, all counters zero."""
    kept, report = cut_utils.drop_malformed(make_tree(**good_event()))
    assert report['kept'] == 1
    assert report['dropped_total'] == 0
    assert len(kept) == 1


# --- reason 1: no candidates -------------------------------------------------

def test_drops_event_with_no_candidates():
    """The most common drop -- ~99% of MC events reconstruct nothing."""
    tree = make_tree(tag_dtf_m=[[], [550.0]], tag_hlt2_tos0=[[], [1.0]],
                     prt_idx_mom=[[], [0, 0]], prt_pid=[[], [-13, 13]])
    kept, report = cut_utils.drop_malformed(tree)
    assert report['dropped_no_candidates'] == 1
    assert report['kept'] == 1
    assert kept['tag_dtf_m'].tolist() == [[550.0]]


def test_no_candidates_is_not_blamed_on_another_reason():
    """An empty event trivially satisfies the other three checks, so it must be
    counted once, under the right heading."""
    tree = make_tree(tag_dtf_m=[[]], tag_hlt2_tos0=[[]],
                     prt_idx_mom=[[]], prt_pid=[[]])
    _, report = cut_utils.drop_malformed(tree)
    assert report['dropped_no_candidates'] == 1
    assert report['dropped_idx_out_of_range'] == 0
    assert report['dropped_daughter_length_mismatch'] == 0
    assert report['dropped_partial_candidate_branch'] == 0


# --- reason 2: index out of range --------------------------------------------

def test_drops_event_whose_index_exceeds_the_candidate_count():
    """prt_idx_mom names candidate 2 but only 2 candidates exist (0 and 1), so
    a daughter points at a candidate that is not there."""
    fields = good_event()
    fields['prt_idx_mom'] = [[0, 0, 1, 2]]
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert report['dropped_idx_out_of_range'] == 1
    assert report['kept'] == 0


def test_index_equal_to_candidate_count_is_out_of_range():
    """Boundary: indices are 0-based, so index == n_tag is one past the end."""
    fields = good_event()
    fields['prt_idx_mom'] = [[0, 0, 1, 2]]        # n_tag == 2, so 2 is invalid
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert report['dropped_idx_out_of_range'] == 1

    fields['prt_idx_mom'] = [[0, 0, 1, 1]]        # 1 is the last valid index
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert report['dropped_idx_out_of_range'] == 0


# --- reason 3: daughter branch length mismatch -------------------------------

def test_drops_event_where_a_daughter_branch_is_short():
    """prt_pid has 2 entries but prt_idx_mom says 4 daughters. This is the fault
    that let a ragged prt_idx_gen through and blew up inside reduce()."""
    fields = good_event()
    fields['prt_pid'] = [[-13, 13]]
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert report['dropped_daughter_length_mismatch'] == 1
    assert report['kept'] == 0


def test_every_prt_branch_is_checked_not_just_the_listed_ones():
    """prt_idx_gen disagreeing with prt_idx_mom about daughter count. Auto-
    derivation is what catches it without anyone having to remember to list the
    branch."""
    fields = good_event()
    fields['prt_idx_gen'] = [[1, 2, 3]]           # 3 entries, not 4
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert 'prt_idx_gen' in report['daughter_fields_checked']
    assert report['dropped_daughter_length_mismatch'] == 1


def test_explicit_daughter_fields_can_check_less():
    """Passing the list explicitly narrows the check -- the documented escape
    hatch. The same tree that fails auto-derivation passes when the offending
    branch is left out."""
    fields = good_event()
    fields['prt_idx_gen'] = [[1, 2, 3]]
    tree = make_tree(**fields)
    _, auto = cut_utils.drop_malformed(tree)
    _, narrow = cut_utils.drop_malformed(tree, daughter_fields=['prt_pid'])
    assert auto['kept'] == 0
    assert narrow['kept'] == 1


# --- reason 4: partial candidate branch --------------------------------------

def test_drops_event_with_a_partial_trigger_branch():
    """2 candidates but only 1 trigger entry: there is no way to know which
    candidate the value belongs to, so the event is unusable."""
    fields = good_event()
    fields['tag_hlt2_tos0'] = [[1.0]]
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert report['dropped_partial_candidate_branch'] == 1
    assert report['kept'] == 0


def test_absent_trigger_branch_is_allowed():
    """An empty trigger branch means the line was never evaluated, which is
    normal -- align() fills it in later. It must NOT be dropped."""
    fields = good_event()
    fields['tag_hlt2_tos0'] = [[]]
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert report['dropped_partial_candidate_branch'] == 0
    assert report['kept'] == 1


def test_partial_by_branch_names_the_offender():
    """The report must say WHICH branch was partial, since that is the thing
    worth chasing upstream in the ntuple production."""
    fields = good_event()
    fields['tag_hlt2_tos0'] = [[1.0]]
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert report['partial_by_branch'] == {'tag_hlt2_tos0': 1}


def test_partial_by_branch_omits_clean_branches():
    """Only offenders are listed, so the dict is empty on good input rather
    than a wall of zeros."""
    _, report = cut_utils.drop_malformed(make_tree(**good_event()))
    assert report['partial_by_branch'] == {}


# --- prefix derivation -------------------------------------------------------

def test_field_lists_are_derived_from_branch_prefixes():
    """prt_* are daughter fields, tag_* are candidate fields, worked out from
    the names of idx_branch and tag_ref rather than passed in."""
    fields = good_event()
    fields['prt_idx_gen'] = [[1, 2, 3, 4]]
    fields['tag_dtf_chi2'] = [[17.0, 14.0]]
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert set(report['daughter_fields_checked']) == {
        'prt_idx_mom', 'prt_pid', 'prt_idx_gen'}
    assert set(report['candidate_fields_checked']) == {
        'tag_dtf_m', 'tag_hlt2_tos0', 'tag_dtf_chi2'}


def test_other_collections_are_not_checked():
    """mc_* and pvr_* live at their own levels and must not be measured against
    the daughter or candidate counts."""
    fields = good_event()
    fields['mc_pid'] = [[221, -13, 13]]           # 3 entries, matches nothing
    fields['pvr_z'] = [[1.0]]
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert report['kept'] == 1
    assert not any(f.startswith('mc_') for f in report['daughter_fields_checked'])
    assert not any(f.startswith('mc_') for f in report['candidate_fields_checked'])


def test_custom_tag_ref_and_idx_branch():
    """A different analysis with different branch names must work by argument
    rather than by editing the function."""
    tree = make_tree(sig_mass=[[550.0]], trk_idx_mom=[[0, 0]], trk_pid=[[-13, 13]])
    kept, report = cut_utils.drop_malformed(tree, tag_ref='sig_mass',
                                            idx_branch='trk_idx_mom')
    assert report['kept'] == 1
    assert set(report['daughter_fields_checked']) == {'trk_idx_mom', 'trk_pid'}
    assert report['candidate_fields_checked'] == ['sig_mass']


# --- report arithmetic and multiple faults -----------------------------------

def test_report_counts_are_self_consistent():
    """read == kept + dropped_total, on a mixed input."""
    tree = make_tree(
        tag_dtf_m=[[550.0, 549.0], [], [550.0], [550.0, 549.0]],
        tag_hlt2_tos0=[[1.0, 0.0], [], [1.0], [1.0]],
        prt_idx_mom=[[0, 0, 1, 1], [], [0, 0], [0, 0, 1, 1]],
        prt_pid=[[-13, 13, -13, 13], [], [-13, 13], [-13, 13, -13, 13]],
    )
    _, report = cut_utils.drop_malformed(tree)
    assert report['read'] == 4
    assert report['read'] == report['kept'] + report['dropped_total']
    assert report['dropped_no_candidates'] == 1          # event 1
    assert report['dropped_partial_candidate_branch'] == 1  # event 3
    assert report['kept'] == 2                            # events 0 and 2


def test_one_event_can_have_two_faults():
    """The per-reason counters may overlap -- they are diagnostics, not a
    partition -- but the event is dropped exactly once."""
    fields = good_event()
    fields['prt_pid'] = [[-13, 13]]        # length mismatch
    fields['tag_hlt2_tos0'] = [[1.0]]      # partial trigger
    _, report = cut_utils.drop_malformed(make_tree(**fields))
    assert report['dropped_daughter_length_mismatch'] == 1
    assert report['dropped_partial_candidate_branch'] == 1
    assert report['dropped_total'] == 1


def test_dropped_counters_exclude_empty_events():
    """The three fault counters are gated on has_candidates so an empty event
    is only ever blamed on 'no candidates'."""
    tree = make_tree(tag_dtf_m=[[], []], tag_hlt2_tos0=[[], []],
                     prt_idx_mom=[[], []], prt_pid=[[], []])
    _, report = cut_utils.drop_malformed(tree)
    assert report['dropped_no_candidates'] == 2
    assert report['dropped_daughter_length_mismatch'] == 0
    assert report['dropped_partial_candidate_branch'] == 0


def test_surviving_tree_is_actually_filtered():
    """The returned tree, not just the report, must have the bad events gone --
    and every branch must be filtered together."""
    tree = make_tree(
        tag_dtf_m=[[550.0], [], [549.0]],
        tag_hlt2_tos0=[[1.0], [], [0.0]],
        prt_idx_mom=[[0, 0], [], [0, 0]],
        prt_pid=[[-13, 13], [], [-13, 13]],
    )
    kept, _ = cut_utils.drop_malformed(tree)
    assert len(kept) == 2
    assert kept['tag_dtf_m'].tolist() == [[550.0], [549.0]]
    assert kept['prt_pid'].tolist() == [[-13.0, 13.0], [-13.0, 13.0]]


def test_report_is_a_plain_dict_of_python_ints():
    """It gets written into a results JSON, so numpy scalars would break
    json.dump."""
    import json
    _, report = cut_utils.drop_malformed(make_tree(**good_event()))
    json.dumps(report)      # raises TypeError if any value is a numpy scalar
    assert isinstance(report['kept'], int)
