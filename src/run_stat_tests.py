"""CLI entry for statistical report generation."""
from src.config import OOF_DIR, REPORTS_DIR
from src.stat_tests import generate_statistical_report

if __name__ == "__main__":
    generate_statistical_report(
        REPORTS_DIR / "experiment_log.csv",
        OOF_DIR,
        REPORTS_DIR / "statistical_comparison.md",
    )
    print("Wrote statistical_comparison.md")
