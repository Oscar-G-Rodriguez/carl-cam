"""Optional, non-recording practice of the real camera control workflow.

This is instruction, not a model evaluation or a new activation gate. Only
completed live controller actions advance the hand version. Desktop controls
have an equivalent version when learned commands are disabled.
"""

from dataclasses import dataclass

from carl_cam.storage import valid_assignments

POSE_WORDS = {"thumbs_up": "thumbs up", "open_palm": "open palm", "ily": "I love you"}


@dataclass(frozen=True, slots=True)
class PracticeStep:
    action: str
    theme: str | None = None


class ControlPractice:
    def __init__(self, assignments, *, hands: bool):
        if not valid_assignments(assignments) or hands and not assignments:
            raise ValueError("Practice needs the user's three distinct filter assignments")
        self.assignments = dict(assignments)
        self.hands = hands
        self.steps = (PracticeStep("frame"),) + tuple(
            step for theme in ("red", "yellow", "blue")
            for step in (PracticeStep("release"), PracticeStep("select", theme), PracticeStep("frame", theme))
        ) + (PracticeStep("menu"), PracticeStep("menu_select", "yellow"), PracticeStep("frame", "yellow"))
        self.index = 0

    @property
    def complete(self):
        return self.index == len(self.steps)

    @property
    def instruction(self):
        if self.complete:
            return "Practice complete. Both switching paths are ready to use. This was not a model accuracy test and saved no recordings."
        step = self.steps[self.index]
        if step.action == "release":
            return "Release the frame and relax your hands. Wait until the effect clears before giving a direct command." if self.hands else "Press Clear frame (or Escape in the preview). Changing filters never fills the whole camera."
        if step.action == "select":
            if not self.hands:
                return f"Choose {step.theme.title()} in the Selected filter control, with no frame visible."
            pose = next(pose for pose, theme in self.assignments.items() if theme == step.theme)
            return f"Hold {POSE_WORDS[pose]} to select {step.theme.title()}. Stay out of the frame pose; selection alone shows no full-image effect."
        if step.action == "menu":
            return "Hold a peace sign to open the head-following menu. Opening it clears the frame. You may then release the peace sign." if self.hands else "Press Open menu. The frame clears and the menu appears over the plain camera."
        if step.action == "menu_select":
            return "Aim your index fingertip at Yellow. Hover freezes the menu; pinch down and release over the same choice. It selects once and closes." if self.hands else "Select Yellow in the menu with a click, or use arrow keys and Enter. Back or Escape cancels without changing filters."
        theme = f" {step.theme.title()}" if step.theme else " the selected filter"
        return f"Form two opposing L-hands and hold a valid four-corner frame to display{theme}." if self.hands else f"Press Mouse frame to display{theme}, then try dragging a corner or using 1–4 and arrow keys."

    def observe(self, *, mode, quad, theme, command=None, manual=False, live=True):
        if self.complete or not live:
            return False
        step = self.steps[self.index]
        if step.action == "release":
            ready = mode == "camera" and quad is None
        elif step.action == "frame":
            ready = quad is not None and (manual if not self.hands else mode == "framed" and command == "frame")
            ready = ready and (step.theme is None or theme == step.theme)
        elif step.action == "select":
            ready = mode == "camera" and quad is None and command == f"select:{step.theme}"
        elif step.action == "menu":
            ready = mode == "menu" and command == "menu"
        else:
            ready = mode == "camera" and quad is None and command == step.theme
        if ready:
            self.index += 1
        return bool(ready)
