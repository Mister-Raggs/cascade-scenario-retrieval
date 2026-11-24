"""Parser structure trees and clip-level atom matching; synthetic labels only."""
import unittest

from baseline.query_parser import translate, Unparsed
from baseline.query_struct import tree, atoms, atom_present


class QueryStructTests(unittest.TestCase):
    def test_temporal_relations_and_atoms(self):
        t = tree(translate('It is only after a bus stops or is stopped but is not parked that the ego vehicle changes to the left lane.'))
        self.assertEqual(t[0], 'before')
        self.assertEqual(atoms(t), ['agent|action=stop|type=bus', 'ego|action=change_lane_left'])
        t = tree(translate('At the same time as a bus turns left, an adult pedestrian is visible.'))
        self.assertEqual(t[0], 'while')
        self.assertEqual(sorted(atoms(t)), ['agent|action=turn_left|type=bus', 'agent|type=ped_adult'])
        with self.assertRaises(Unparsed):
            translate('A unicorn juggles on the median.')

    def test_atom_matching_requires_same_agent_and_respects_aliases(self):
        labels = {'agent:oxd:Car', 'agent_act:oxd:Car|oxd:MakeARightTurn (unprotected)', 'agent:Pedestrian (Adult)',
                  'agent_act:Pedestrian (Adult)|oxd:Walk', 'ego_act:oxd:ChangeLane (left)', 'obj:oxd:TrafficCone', 'light:Red'}
        self.assertTrue(atom_present('agent|action=turn_right|type=car', labels))
        self.assertFalse(atom_present('agent|action=walk|type=car', labels))
        self.assertTrue(atom_present('agent|action=walk|type=ped', labels))
        self.assertTrue(atom_present('ego|action=change_lane_left', labels))
        self.assertFalse(atom_present('ego|action=change_lane_right', labels))
        self.assertTrue(atom_present('obj|type=bollard,cone', labels))
        self.assertTrue(atom_present('light|color=red', labels)); self.assertFalse(atom_present('light|color=green', labels))


if __name__ == '__main__':
    unittest.main()


class ProviderAlignmentTests(unittest.TestCase):
    def test_development_only_features_align_with_zero_rows_and_mask(self):
        import numpy as np
        from scripts.structured_eval import _aligned
        out, mask = _aligned([[1, 2], [3, 4]], ['b', 'a'], ['a', 'audit', 'b'])
        self.assertEqual(out.tolist(), [[3, 4], [0, 0], [1, 2]])
        self.assertEqual(mask.tolist(), [True, False, True])
