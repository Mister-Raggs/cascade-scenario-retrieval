"""Natural-language yes/no questions for structure atoms (targeted VLM probing)."""
from __future__ import annotations

TYPES = {
    'animal': 'an animal', 'bicycle': 'a bicycle or cyclist', 'bus': 'a bus', 'car': 'a car', 'emergency': 'an emergency vehicle',
    'motorcycle': 'a motorcycle', 'ped': 'a pedestrian', 'ped_adult': 'an adult pedestrian', 'ped_personnel': 'a road worker',
    'truck': 'a truck or heavy-duty vehicle', 'vehicle': 'another vehicle', 'bollard': 'a traffic cone or bollard',
    'cone': 'a traffic cone or bollard', 'boom_gate': 'a boom gate', 'do_not_enter': 'a do-not-enter sign',
    'other_debris': 'debris on the road', 'portable_indicator': 'a temporary road marker', 'speed_limit_sign': 'a speed-limit sign',
    'stop_sign': 'a stop sign', 'warning_sign': 'a warning sign', 'yield_sign': 'a yield sign',
}
AGENT_ACTIONS = {
    'change_lane': 'change lanes', 'change_lane_left': 'change to the left lane', 'change_lane_right': 'change to the right lane',
    'decel': 'slow down', 'drive': 'drive in its lane', 'follow': 'follow another road user', 'jaywalk_action': 'jaywalk across the road',
    'nudge': 'make a small sideways move', 'nudge_in': 'move slightly within its lane', 'nudge_out': 'move slightly out of its lane',
    'park': 'park', 'reverse': 'reverse', 'stand': 'stand still', 'stop': 'stop or stand stopped (not parked)',
    'turn': 'turn', 'turn_left': 'turn left', 'turn_right': 'turn right', 'walk': 'walk', 'yield': 'yield to another road user',
}
EGO = 'the camera car (the ego vehicle recording this video)'


def question(key):
    entity, *parts = key.split('|')
    fields = {p.split('=')[0]: p.split('=')[1].split(',') for p in parts}
    if entity == 'ego':
        actions = ' or '.join(AGENT_ACTIONS[a] for a in fields['action'])
        return f'Does {EGO} {actions} at any point in this video?'
    if entity == 'obj':
        return f'Is {TYPES[fields["type"][0]]} visible at any point in this video?'
    if entity == 'light':
        shape = {'arrow_left': ' left-arrow', 'arrow_right': ' right-arrow', 'arrow_up': ' straight-arrow'}.get(fields.get('shape', [''])[0], '')
        return f'Is a {fields["color"][0]}{shape} traffic signal shown at any point in this video?'
    if entity == 'agent':
        who = ' or '.join(TYPES[t] for t in fields['type']) if 'type' in fields else 'any road user other than the camera car'
        if 'action' not in fields:
            return f'Is {who} visible at any point in this video?'
        actions = ' or '.join(AGENT_ACTIONS[a] for a in fields['action'])
        return f'Does {who} {actions} at any point in this video?'
    raise ValueError(key)
