"""
Sleep staging panel – displays the hypnogram and EEG waveform.
Uses pyqtgraph for fast, interactive plotting.
"""

import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QGroupBox

from utils.logger import logger

try:
    import pyqtgraph as pg
    HAS_PG = True
except ImportError:
    HAS_PG = False
    logger.warning("pyqtgraph not installed – EEG plots will be unavailable.")


class SleepStagePanel(QWidget):
    """Panel showing EEG signal waveform and hypnogram."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- EEG waveform ---
        eeg_grp = QGroupBox("EEG Signal")
        eeg_grp.setStyleSheet("QGroupBox { font-weight: bold; color: #1565C0; }")
        eeg_layout = QVBoxLayout(eeg_grp)

        if HAS_PG:
            self.eeg_plot = pg.PlotWidget(title="EEG Waveform")
            self.eeg_plot.setLabel("bottom", "Time", units="s")
            self.eeg_plot.setLabel("left", "Amplitude", units="µV")
            self.eeg_plot.showGrid(x=True, y=True, alpha=0.3)
            eeg_layout.addWidget(self.eeg_plot)
        else:
            eeg_layout.addWidget(QLabel("pyqtgraph not available."))
            self.eeg_plot = None

        layout.addWidget(eeg_grp)

        # --- Hypnogram ---
        hyp_grp = QGroupBox("Hypnogram")
        hyp_grp.setStyleSheet("QGroupBox { font-weight: bold; color: #1565C0; }")
        hyp_layout = QVBoxLayout(hyp_grp)

        if HAS_PG:
            self.hyp_plot = pg.PlotWidget(title="Sleep Stages")
            self.hyp_plot.setLabel("bottom", "Epoch (30 s)")
            self.hyp_plot.setLabel("left", "Stage")
            self.hyp_plot.showGrid(x=True, y=True, alpha=0.3)
            # Custom Y-axis ticks for stage names
            ay = self.hyp_plot.getAxis("left")
            ay.setTicks([[(0, "W"), (1, "N1"), (2, "N2"), (3, "N3"), (4, "R")]])
            hyp_layout.addWidget(self.hyp_plot)
        else:
            hyp_layout.addWidget(QLabel("pyqtgraph not available."))
            self.hyp_plot = None

        layout.addWidget(hyp_grp)

    # ---- Public API ----

    def plot_eeg(self, data: np.ndarray, sfreq: float) -> None:
        """Plot raw EEG data (single channel, in µV)."""
        if self.eeg_plot is None:
            return
        self.eeg_plot.clear()
        times = np.arange(len(data)) / sfreq
        # Down-sample for performance if very long
        max_pts = 500_000
        if len(data) > max_pts:
            step = len(data) // max_pts
            data = data[::step]
            times = times[::step]
        self.eeg_plot.plot(times, data, pen=pg.mkPen("#1565C0", width=1))

    def plot_hypnogram(self, hypnogram: np.ndarray) -> None:
        """Plot the hypnogram (integer array, one per 30-s epoch)."""
        if self.hyp_plot is None:
            return
        self.hyp_plot.clear()
        x = np.arange(len(hypnogram))
        self.hyp_plot.plot(x, hypnogram, pen=pg.mkPen("#E65100", width=2),
                           stepMode="right", fillLevel=-0.5,
                           brush=pg.mkBrush(230, 81, 0, 40))

    def plot_slow_waves(self, times: list[float], amplitudes: list[float]) -> None:
        """Overlay slow-wave markers on the EEG plot."""
        if self.eeg_plot is None or not times:
            return
        scatter = pg.ScatterPlotItem(
            x=times, y=[-a for a in amplitudes],
            pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 180),
            size=6, symbol="o",
        )
        self.eeg_plot.addItem(scatter)

    def plot_delta_band(self, data: np.ndarray, sfreq: float) -> None:
        """Overlay delta-band filtered signal on the EEG plot."""
        if self.eeg_plot is None:
            return
        times = np.arange(len(data)) / sfreq
        max_pts = 500_000
        if len(data) > max_pts:
            step = len(data) // max_pts
            data = data[::step]
            times = times[::step]
        self.eeg_plot.plot(times, data, pen=pg.mkPen("#43A047", width=1.2))
