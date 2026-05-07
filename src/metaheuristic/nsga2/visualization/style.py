import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ── Font setup ──────────────────────────────────────────────────
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except (RuntimeError, FileNotFoundError, OSError):
    pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Per-metric style ───────────────────────────────────────────
METRIC_STYLE = {
    "GD":    {"color": "#2563EB", "marker": "o"},
    "IGD":   {"color": "#F97316", "marker": "s"},
    "IGD+":  {"color": "#8B5CF6", "marker": "D"},
    "Spread": {"color": "#DC2626", "marker": "^"},
    "HV":    {"color": "#10B981", "marker": "p"},
}

# ── Plot defaults ──────────────────────────────────────────────
LABEL_SIZE = 12
TITLE_SIZE = 14
DPI = 200
LINE_WIDTH = 2
MARKER_SIZE = 3
SCATTER_SIZE = 40
ALPHA_GRID = 0.3
