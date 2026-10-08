"""Developer preflight for the exact frozen-build test-package exclusion.

Install the blocker before importing the app or PyTorch. A successful smoke
check exercises real CPU fitting, load, vision and widgets without PyTorch's
own internal test suite. This cannot substitute for the actual frozen check.
"""

import argparse
import importlib.abc
import sys
import traceback
from fnmatch import fnmatchcase
from pathlib import Path

EXCLUSIONS = ("torch.testing._internal.common_methods_invocations", "torch.testing._internal.common_utils",
              "torch.testing._internal.common_device_type", "torch.testing._internal.common_modules",
              "torch.testing._internal.common_nn", "torch.testing._internal.opinfo*", "torch._dynamo.test_case")


class BlockInternalTests(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = set()
        self.callers = set()

    def find_spec(self, fullname, path=None, target=None):
        if any(fnmatchcase(fullname, pattern) for pattern in EXCLUSIONS):
            self.attempts.add(fullname)
            self.callers.update((frame.f_globals.get("__name__", "unknown"), line)
                               for frame, line in traceback.walk_stack(None)
                               if frame.f_globals.get("__name__", "").startswith("torch"))
            raise ModuleNotFoundError(f"Excluded development-only test module: {fullname}", name=fullname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    from carl_cam.vision_privacy import require_verified_vision_runtime
    require_verified_vision_runtime()
    blocker = BlockInternalTests()
    sys.meta_path.insert(0, blocker)
    from carl_cam.selftest import run_selftest
    result = run_selftest(arguments.report)
    print({"excluded_patterns": EXCLUSIONS, "attempted_optional_imports": sorted(blocker.attempts),
           "torch_import_callers": sorted(blocker.callers),
           "preflight_passed": result == 0, "frozen_verification_still_required": True})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
