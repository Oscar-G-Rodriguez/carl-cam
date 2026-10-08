from carl_cam.menu import MenuFollower, Rect, orbit, perimeter


def test_menu_orbits_head_with_large_targets_and_face_clearance():
    face = Rect(540, 190, 160, 210)
    layout = orbit((620, 295), face)
    for name, rect in layout.cards:
        assert rect.width >= 160 and rect.height >= 90
        assert rect.x >= 24 and rect.y >= 24
        assert rect.x + rect.width <= 1256 and rect.y + rect.height <= 696
        assert not rect.overlaps(face, 24)
        assert layout.hit((rect.x + 50, rect.y + 50)) == name


def test_head_follow_freezes_for_hover_and_pinch_then_reattaches():
    follower = MenuFollower()
    first = follower.update((640, 280), Rect(560, 190, 160, 180), 1, hover=False, pinching=False)
    hovered = follower.update((690, 280), Rect(610, 190, 160, 180), 1.1, hover=True, pinching=False)
    assert hovered.cards == first.cards and hovered.frozen
    pinched = follower.update(None, None, 1.5, hover=False, pinching=True)
    assert pinched.cards == first.cards and pinched.frozen
    followed = follower.update((690, 280), Rect(610, 190, 160, 180), 1.6, hover=False, pinching=False)
    assert followed.cards != first.cards and not followed.frozen


def test_head_loss_holds_then_switches_to_fixed_safe_targets():
    follower = MenuFollower()
    first = follower.update((640, 280), Rect(560, 190, 160, 180), 1, hover=False, pinching=False)
    assert follower.update(None, None, 1.2, hover=False, pinching=False).cards == first.cards
    assert follower.update(None, None, 1.31, hover=False, pinching=False).fixed


def test_output_hit_geometry_mirrors_once_without_changing_action_identity():
    layout = orbit((620, 295), Rect(540, 190, 160, 210))
    for name, rect in layout.cards:
        point = (rect.x + 60, rect.y + 50)
        assert layout.output().hit((1280 - point[0], point[1])) == name


def test_impossible_face_clearance_hides_unsafe_cards_and_requests_space():
    face = Rect(0, 0, 1280, 720)
    for layout in (perimeter(face), orbit((640, 360), face)):
        assert layout.fixed and layout.needs_space and not layout.cards
        assert layout.hit((100, 100)) is None
        assert layout.output().needs_space
    follower = MenuFollower()
    tight = follower.update((640, 360), face, 1, hover=False, pinching=False)
    assert tight.needs_space
    assert not follower.update((640, 280), Rect(560, 190, 160, 180), 3, hover=False, pinching=False).needs_space


def test_fixed_perimeter_never_selects_a_face_overlapping_slot():
    for face in (Rect(20, 50, 600, 500), Rect(500, 200, 650, 450), Rect(300, 5, 600, 680)):
        layout = perimeter(face)
        assert all(not rect.overlaps(face, 24) for _, rect in layout.cards)
