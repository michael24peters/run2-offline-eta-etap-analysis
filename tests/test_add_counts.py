"""
Unit tests for cut_utils.add_counts(), the sum of two reports of counts.
"""

import cut_utils


def test_counts_sum_key_by_key_at_every_depth():
    total = {'read': 3, 'triggers': {'L0': {'tag_l0_tos0': 1}}}
    counts = {'read': 2, 'triggers': {'L0': {'tag_l0_tos0': 4}}}
    assert cut_utils.add_counts(total, counts) == {'read': 5,
                                                   'triggers': {'L0': {'tag_l0_tos0': 5}}}


def test_key_in_only_one_report_keeps_its_value():
    total = {'partial_by_branch': {'tag_hlt2_tos0': 1}}
    counts = {'partial_by_branch': {'tag_hlt1_tos0': 2}, 'kept': 7}
    assert cut_utils.add_counts(total, counts) == {
        'partial_by_branch': {'tag_hlt2_tos0': 1, 'tag_hlt1_tos0': 2}, 'kept': 7}


def test_empty_total_keeps_the_order_of_counts():
    counts = {'all candidates': 4, 'TOS up to L0': 3, 'IsMuonTight': 1}
    assert list(cut_utils.add_counts({}, counts)) == ['all candidates', 'TOS up to L0',
                                                      'IsMuonTight']
