from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
OOF_DIR = REPORTS_DIR / "oof_predictions"

RANDOM_STATE = 42
N_SPLITS = 5
POSITIVE_LABEL = "worn"
LABEL_MAP = {"unworn": 0, "worn": 1}

META_COLS = [
    "M1_CURRENT_PROGRAM_NUMBER",
    "M1_sequence_number",
    "M1_CURRENT_FEEDRATE",
    "Machining_Process",
]

PHASES = [
    "Layer 1 Up",
    "Layer 1 Down",
    "Layer 2 Up",
    "Layer 2 Down",
    "Layer 3 Up",
    "Layer 3 Down",
]

DOC_RULE = {
    "feedrate_bad": ("M1_CURRENT_FEEDRATE", 50.0),
    "x_position_bad": ("X1_ActualPosition", 198.0),
    "program_bad": ("M1_CURRENT_PROGRAM_NUMBER", 0),
}
