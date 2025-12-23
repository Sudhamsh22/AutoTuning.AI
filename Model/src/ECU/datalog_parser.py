from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd


TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3},")


@dataclass
class DataLogParseResult:
    df: pd.DataFrame
    channels: list[str]


def parse_ecu_manager_datalog(path: str | Path) -> DataLogParseResult:
    """
    Parses ECU Manager DataLog format:
      - metadata lines
      - repeated "Channel : <name>" blocks
      - data rows start with "HH:MM:SS.mmm,"
    """
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()

    channels: list[str] = []
    for line in lines:
        if line.startswith("Channel :"):
            channels.append(line.split("Channel :")[1].strip())

    if not channels:
        raise ValueError(f"No channels found in file: {path}")

    # Data rows: time + len(channels) values
    rows = []
    for line in lines:
        if TIME_RE.match(line):
            parts = line.split(",")
            # pad missing values
            expected = 1 + len(channels)
            if len(parts) < expected:
                parts += [""] * (expected - len(parts))
            rows.append(parts[:expected])

    if not rows:
        raise ValueError(f"No data rows found in file: {path}")

    cols = ["Time"] + channels
    df = pd.DataFrame(rows, columns=cols)

    # Convert numeric columns (everything except Time)
    for c in channels:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return DataLogParseResult(df=df, channels=channels)
