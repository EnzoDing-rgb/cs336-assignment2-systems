"""Entry: `uv run python -m cs336_systems.nsys_profile ab|cde ...`"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"ab", "cde"}:
        print(
            "usage: python -m cs336_systems.nsys_profile ab|cde <subcommand>...\n"
            "  ab   parts (a)(b)\n"
            "  cde  parts (c)(d)(e)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    which = sys.argv.pop(1)
    if which == "ab":
        from cs336_systems.nsys_profile.ab import main as ab_main

        ab_main()
    else:
        from cs336_systems.nsys_profile.cde import main as cde_main

        cde_main()


if __name__ == "__main__":
    main()
