from test_features import hand

from carl_cam.pointer import HandPointer


def test_pointer_keeps_opening_hand_even_when_another_hand_is_larger():
    owner = hand(label="Left")
    other = hand(offset=(850, 350), scale=2, label="Right")
    pointer = HandPointer()
    pointer.begin(owner, 1)
    result = pointer.update((other, owner), 1.05)
    assert result.position == owner.landmarks[8][:2] and not result.owner_changed
    assert pointer.update((other,), 1.2) is None
    replacement = pointer.update((other,), 1.26)
    assert replacement.owner_changed and replacement.position == other.landmarks[8][:2]


def test_uncertain_tracking_does_not_create_pointer_coordinates():
    pointer = HandPointer()
    assert pointer.update((), 1) is None
    assert pointer.update((hand(scale=0.1),), 2) is None
