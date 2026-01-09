"""Compatibility entrypoint for the Pluto Spectrum Analyzer.

Delegates to the packaged application entrypoint. This file should not contain
UI, DSP, or SDR logic.
"""

from pluto_spectrum_analyzer.app import main


if __name__ == "__main__":
    raise SystemExit(main())
