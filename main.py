#!/usr/bin/env python3
"""Compat shim for running Multivac as a standalone script."""

from multivac.cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
