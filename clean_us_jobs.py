import re, pandas as pd

INPUT  = "job_descriptions.csv"
OUTPUT = "job_descriptions_us.csv"

US_REGEX = re.compile(
    r"\b("
    r"united\s+states(?:\s+of\s+america)?|"  # United States or United States of America
    r"usa|"                                 # USA
    r"u\.s\.a\.?|"                          # U.S.A. or U.S.A
    r"u\.s\.?|"                             # U.S. or U.S
    r"\bus\b"                               # plain 'US' word
    r")\b",
    re.IGNORECASE
)

def is_us_country(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return bool(US_REGEX.search(s))

def is_us_country(s: str) -> bool:
    """Return True if text matches U.S. pattern."""
    if not isinstance(s, str):
        return False
    return bool(US_REGEX.search(s))

def main():
    chunksize = 200_000
    wrote_header = False
    total_in = total_kept = 0

    for chunk in pd.read_csv(INPUT, dtype=str, keep_default_na=False, chunksize=chunksize):
        if "Country" in chunk.columns:
            mask = chunk["Country"].apply(is_us_country)
        else:
            if "location" in chunk.columns:
                mask = chunk["location"].apply(is_us_country)
            else:
                mask = pd.Series(False, index=chunk.index)

        kept = chunk[mask]
        total_in += len(chunk)
        total_kept += len(kept)

        # append to output CSV
        kept.to_csv(OUTPUT, index=False, mode="a", header=(not wrote_header))
        wrote_header = True

    print(f"[DONE] Read {total_in:,} rows → kept {total_kept:,} US rows → {OUTPUT}")

if __name__ == "__main__":
    main()