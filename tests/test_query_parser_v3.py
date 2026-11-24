"""Parser v3 only adds coverage: v2 translations are unchanged; synthetic sentences in the released query style."""
import unittest

from baseline.query_parser import Unparsed
from baseline.query_parser_v2 import translate as v2
from baseline.query_parser_v3 import translate as v3
from baseline.query_parser_v4 import translate as v4
from baseline.query_struct import atoms, tree


def parsed(fn, text):
    try:
        return sorted(set(atoms(tree(fn(text)))))
    except Unparsed:
        return None


class ParserV3Tests(unittest.TestCase):
    def test_v2_parses_are_unchanged(self):
        for text in ('A car turns left.', 'After a road user stops, the ego vehicle stops.', 'The ego vehicle yields since a bus turns left.'):
            self.assertIsNotNone(parsed(v2, text))
            self.assertEqual(v3(text), v2(text))

    def test_new_frames_map_to_cause_before_effect(self):
        cases = {
            'A car turns right; some time afterward, the ego vehicle stops.': ['agent|action=turn_right|type=car', 'ego|action=stop'],
            'Because a bus turns left, the ego vehicle yields.': ['agent|action=turn_left|type=bus', 'ego|action=yield'],
            'A bus turns left; consequently, the ego vehicle stops.': ['agent|action=turn_left|type=bus', 'ego|action=stop'],
            'It is because a stop sign is present that a road user yields.': ['agent|action=yield', 'obj|type=stop_sign'],
            'Here, a bus is decelerating.': ['agent|action=decel|type=bus'],
        }
        for text, want in cases.items():
            self.assertIsNone(parsed(v2, text), text)
            self.assertEqual(parsed(v3, text), want, text)
        self.assertEqual(v3('Because a bus turns left, the ego vehicle yields.'), v2('After a bus turns left, the ego vehicle yields.'))

    def test_still_unparsed_stays_unparsed(self):
        with self.assertRaises(Unparsed):
            v3('While a bus drives in its lane, the same bus is signaling.')


class ParserV4Tests(unittest.TestCase):
    def test_time_limits_keep_the_sequence(self):
        want = ['agent|action=change_lane_left', 'ego|action=change_lane']
        for text in ('The ego vehicle finishes changing lanes, and within 7 seconds of that, a road user starts to change to the left lane.',
                     'A road user starts to change to the left lane no more than 7 seconds after the ego vehicle finishes changing lanes.'):
            self.assertIsNone(parsed(v3, text))
            self.assertEqual(parsed(v4, text), want)
            self.assertEqual(tree(v4(text))[0], 'before')
        self.assertEqual(v4('A car turns left.'), v3('A car turns left.'))


if __name__ == '__main__':
    unittest.main()
