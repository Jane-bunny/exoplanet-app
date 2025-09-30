from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

# ---------- IO helpers ----------

READ_OPTS = dict(
    engine="python",      # required for sep=None
    sep=None,             # sniff delimiter
    comment="#",
    na_values=["", "NaN", "--", " "],
)

def read_catalog(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, **READ_OPTS)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------- Cleaning: Kepler ----------

KEPLER_COLS = {
    "kepid": "star_id",
    "kepoi_name": "planet_id",
    "koi_disposition": "disposition",
    "koi_period": "orbital_period",
    "koi_duration": "transit_duration",
    "koi_depth": "transit_depth",
    "koi_prad": "planet_radius",
    "koi_teq": "equilibrium_temp",
    "koi_insol": "insolation_flux",
    "koi_model_snr": "snr",
    "koi_steff": "stellar_teff",
    "koi_slogg": "stellar_logg",
    "koi_srad": "stellar_radius",
    "ra": "ra",
    "dec": "dec",
    "koi_kepmag": "stellar_mag",
}

def clean_kepler_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in KEPLER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kepler input missing expected columns: {missing}")
    out = df[list(KEPLER_COLS.keys())].rename(columns=KEPLER_COLS).copy()
    return out

# ---------- Cleaning: TESS ----------

TESS_COLS = {
    "toi": "planet_id",
    "tid": "star_id",
    "tfopwg_disp": "disposition",
    "pl_orbper": "orbital_period",
    "pl_trandurh": "transit_duration",
    "pl_trandep": "transit_depth",
    "pl_rade": "planet_radius",
    "pl_eqt": "equilibrium_temp",
    "pl_insol": "insolation_flux",
    "st_teff": "stellar_teff",
    "st_logg": "stellar_logg",
    "st_rad": "stellar_radius",
    "ra": "ra",
    "dec": "dec",
    "st_tmag": "stellar_mag",
    "st_dist": "stellar_dist",
}

# mapping to align with Kepler labels
TESS_DISP_MAP = {
    "PC": "CANDIDATE",
    "APC": "CANDIDATE",
    "CP": "CONFIRMED",
    "FP": "FALSE POSITIVE",
    "FA": "FALSE POSITIVE",
    # "KP" excluded to avoid leakage (Known Planet, often from Kepler/K2)
}

def clean_tess_df(df: pd.DataFrame, drop_known_planets: bool = True) -> pd.DataFrame:
    missing = [c for c in TESS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"TESS input missing expected columns: {missing}")
    out = df[list(TESS_COLS.keys())].rename(columns=TESS_COLS).copy()

    if drop_known_planets and "disposition" in out.columns:
        # exclude KP before mapping to avoid leakage
        out = out[out["disposition"] != "KP"].copy()

    # aligned label
    out["disposition_aligned"] = out["disposition"].map(TESS_DISP_MAP)
    return out

# ---------- CLI / script entry ----------

def main():
    parser = argparse.ArgumentParser(description="Clean Kepler & TESS catalogs to a common schema.")
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to data/raw")
    parser.add_argument("--processed-dir", type=str, default=None, help="Path to data/processed")
    parser.add_argument("--kepler-file", type=str, default=None, help="Kepler CSV filename in raw-dir")
    parser.add_argument("--tess-file", type=str, default=None, help="TESS CSV filename in raw-dir")
    parser.add_argument("--no-drop-kp", action="store_true", help="Do NOT drop TESS KP (Known Planet)")
    args = parser.parse_args()

    # Resolve project root as parent of this file's parent (…/src/ -> project root)
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = Path(args.raw_dir) if args.raw_dir else project_root / "data" / "raw"
    processed_dir = Path(args.processed_dir) if args.processed_dir else project_root / "data" / "processed"
    ensure_dir(processed_dir)

    # Default filenames if not provided
    if args.kepler_file is None:
        # pick first file that looks like the cumulative KOI export
        candidates = sorted(raw_dir.glob("*cumulative*.csv"))
        if not candidates:
            raise FileNotFoundError("Kepler file not provided and no '*cumulative*.csv' found in raw-dir.")
        kepler_path = candidates[0]
    else:
        kepler_path = raw_dir / args.kepler_file

    if args.tess_file is None:
        candidates = sorted(raw_dir.glob("*TOI*.csv"))
        if not candidates:
            raise FileNotFoundError("TESS file not provided and no '*TOI*.csv' found in raw-dir.")
        tess_path = candidates[0]
    else:
        tess_path = raw_dir / args.tess_file

    # Read
    df_kepler = read_catalog(kepler_path)
    df_tess = read_catalog(tess_path)

    # Clean
    df_kepler_clean = clean_kepler_df(df_kepler)
    df_tess_clean = clean_tess_df(df_tess, drop_known_planets=not args.no_drop_kp)

    # Save
    kepler_out = processed_dir / "kepler_clean.csv"
    tess_out = processed_dir / "tess_clean.csv"
    df_kepler_clean.to_csv(kepler_out, index=False)
    df_tess_clean.to_csv(tess_out, index=False)

    # Simple report
    print(f"[Kepler] {kepler_path.name} -> {kepler_out.name}  shape={df_kepler_clean.shape}")
    print(f"[TESS]   {tess_path.name} -> {tess_out.name}    shape={df_tess_clean.shape}")
    if "disposition" in df_kepler_clean:
        print("\nKepler dispositions:")
        print(df_kepler_clean["disposition"].value_counts())
    if "disposition_aligned" in df_tess_clean:
        print("\nTESS dispositions (aligned):")
        print(df_tess_clean["disposition_aligned"].value_counts())

if __name__ == "__main__":
    main()