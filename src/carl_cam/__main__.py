"""Compose the desktop application without owning filter or learning logic."""

from carl_cam.ui import run


def main() -> int:
    import sys
    if "--self-test" in sys.argv:
        import argparse
        from pathlib import Path

        from carl_cam.selftest import run_selftest
        parser = argparse.ArgumentParser(description="Developer-only offline bundle check")
        parser.add_argument("--self-test", action="store_true")
        parser.add_argument("--self-test-report", type=Path, required=True)
        arguments = parser.parse_args()
        return run_selftest(arguments.self_test_report)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
