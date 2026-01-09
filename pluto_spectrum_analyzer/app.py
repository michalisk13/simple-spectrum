"""Application entrypoint wiring for the spectrum analyzer.

Creates the Qt application, config, and main window. This module must not
contain UI or SDR logic beyond orchestration.
"""

import sys

from pyqtgraph.Qt import QtWidgets

from pluto_spectrum_analyzer.config import SpectrumConfig
from pluto_spectrum_analyzer.ui.main_window import SpectrumWindow


def main() -> int:
    cfg = SpectrumConfig()
    app = QtWidgets.QApplication(sys.argv)
    window = SpectrumWindow(cfg)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
