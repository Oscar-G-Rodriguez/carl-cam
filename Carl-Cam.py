"""Windows deployment entry point; runtime services remain in carl_cam."""

from carl_cam.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
