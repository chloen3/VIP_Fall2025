import json, re
import pandas as pd

TOP_K = 5
MIN_SKILL_HITS = 0

WORD_SPLIT = re.compile(r"[^\w+#.]+")

def to_list_lower(x):
    if not x: return []
    if isinstance(x, list): 
        return [str(s).strip().lower() for s in x]
    return [str(x).strip().lower()]

def tokenize(text: str):
    if pd.isna(text) or text is None: 
        return []
    return [t for t in WORD_SPLIT.split(str(text).lower()) if t]

def contains_any(haystack: str, needles):
    h = haystack.lower()
    return any(n in h for n in needles if n)

def count_skill_hits(text: str, skills):
    # count unique skill keywords present
    text_l = text.lower()
    hits = 0
    seen = set()
    for s in skills:
        if s and s not in seen and s in text_l:
            hits += 1
            seen.add(s)
    return hits

def main(jobs_csv: str, users_json: str):
    with open(users_json) as f:
        users = json.load(f)["users"]

    # 1) read CSV
    df = pd.read_csv(jobs_csv, dtype=str, keep_default_na=False, engine="c")

    # 2) drop unneeded columns
    want_lower = {
        "job id", "job title", "role", "location", "country",
        "work type", "job description", "skills"
    }
    keep_cols = [c for c in df.columns if c.strip().lower() in want_lower]
    df = df[keep_cols]

    # 3) lowercase col headers
    df.columns = [c.strip().lower() for c in df.columns]

    def col(name):
        return df[name] if name in df.columns else pd.Series([""] * len(df), index=df.index)

    # colu references
    title   = col("job title")
    role    = col("role")
    desc    = col("job description")
    loc     = col("location")
    country = col("country")
    wtype   = col("work type")
    jskills = col("skills")

    title_l   = title.str.lower().fillna("")
    role_l    = role.str.lower().fillna("")
    desc_l    = desc.str.lower().fillna("")
    loc_l     = loc.str.lower().fillna("")
    country_l = country.str.lower().fillna("")
    wtype_l   = wtype.str.lower().fillna("")
    jskills_l = jskills.str.lower().fillna("")

    for u in users:
        name = u.get("name", u.get("user_id","(user)"))
        desired_roles = [s.strip().lower() for s in u.get("desired_roles", []) if s]
        skills = [s.strip().lower() for s in u.get("skills", []) if s]
        loc_keywords = [s.strip().lower() for s in u.get("location_keywords", []) if s]
        remote_ok = bool(u.get("remote_ok", True))

        # role mask: any desired role in title OR role OR desc
        if desired_roles:
            role_mask = pd.Series(False, index=df.index)
            for term in desired_roles:
                if term:
                    role_mask |= title_l.str.contains(re.escape(term), na=False)
                    role_mask |= role_l.str.contains(re.escape(term), na=False)
                    role_mask |= desc_l.str.contains(re.escape(term), na=False)
        else:
            role_mask = pd.Series(False, index=df.index)

        # location mask: any loc keyword in location/country/work type/desc
        if loc_keywords:
            loc_mask = pd.Series(False, index=df.index)
            for term in loc_keywords:
                if term:
                    loc_mask |= loc_l.str.contains(re.escape(term), na=False)
                    loc_mask |= country_l.str.contains(re.escape(term), na=False)
                    loc_mask |= wtype_l.str.contains(re.escape(term), na=False)
                    loc_mask |= desc_l.str.contains(re.escape(term), na=False)
        else:
            loc_mask = pd.Series(False, index=df.index)

        # remote allowed adds simple "remote" signal
        if remote_ok:
            loc_mask |= (loc_l.str.contains("remote", regex=False, na=False) |
                         country_l.str.contains("remote", regex=False, na=False) |
                         wtype_l.str.contains("remote", regex=False, na=False) |
                         desc_l.str.contains("remote", regex=False, na=False))

        # candidate rows are any with role OR location hit
        cand_idx = df.index[role_mask | loc_mask]
        if len(cand_idx) == 0:
            print("="*80)
            print(f"Top 0 matches for: {name}")
            print("="*80 + "\n")
            continue

        # 6) build blobs ONLY for the candidates (for your existing scoring)
        title_c   = title_l.loc[cand_idx]
        role_c    = role_l.loc[cand_idx]
        desc_c    = desc_l.loc[cand_idx]
        loc_c     = loc_l.loc[cand_idx]
        country_c = country_l.loc[cand_idx]
        wtype_c   = wtype_l.loc[cand_idx]
        jskills_c = jskills_l.loc[cand_idx]

        blob_series = (title_c + " " + role_c + " " + desc_c + " " + jskills_c)
        loc_series  = (loc_c + " " + country_c + " " + wtype_c + " " + desc_c)

        scores = []
        for idx in cand_idx:
            jb_blob = blob_series.loc[idx]
            lc_blob = loc_series.loc[idx]

            # skills
            text_l = jb_blob  # already lowercased
            hits = 0
            if skills:
                seen = set()
                for s in skills:
                    if s and s not in seen and s in text_l:
                        hits += 1
                        seen.add(s)
            skill_hits = hits

            # role hit (we already have role_mask; re-derive per row to keep your structure)
            role_hit = 1 if role_mask.loc[idx] else 0

            # loc hit (same)
            loc_hit = 1 if loc_mask.loc[idx] else 0

            score = skill_hits + role_hit + loc_hit
            if skill_hits < MIN_SKILL_HITS:
                continue

            scores.append((score, idx, skill_hits, role_hit, loc_hit))

        scores.sort(key=lambda t: (t[0], t[2], t[3], t[4]), reverse=True)
        top = scores[:TOP_K]

        print("="*80)
        print(f"Top {len(top)} matches for: {name}")
        print("="*80)

        for rank, (score, idx, s_hits, r_hit, l_hit) in enumerate(top, 1):
            jt = (title.loc[idx] if isinstance(title, pd.Series) else "") or ""
            rl = (role.loc[idx] if isinstance(role, pd.Series) else "") or ""
            lc = (loc.loc[idx] if isinstance(loc, pd.Series) else "") or ""
            cn = (country.loc[idx] if isinstance(country, pd.Series) else "") or ""
            wt = (wtype.loc[idx] if isinstance(wtype, pd.Series) else "") or ""
            jid = df.loc[idx, "job id"] if "job id" in df.columns else "(n/a)"
            print(f"{rank:>2}. {jt or '(no title)'}")
            print(f"    Job Id: {jid}")
            print(f"    Role: {rl} | Location: {lc}, {cn} | Work Type: {wt}")
            print(f"    Score: {score}  (skills:{s_hits}  role:{bool(r_hit)}  loc/remote:{bool(l_hit)})")
            print("-"*80)
        print()



if __name__ == "__main__":
    JOBS_CSV = "./job_descriptions.csv"
    USERS_JSON = "./user_profiles.json"
    main(JOBS_CSV, USERS_JSON)