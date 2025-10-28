import json, re, math
import pandas as pd

WEIGHTS = dict(
    skill=1.0,
    role=1.0,
    location=1.0,
    exp=1.0,
    salary=1.0,
    wtype=0.6,
    csize=0.5,
    tags=0.4,
    geo=1.2,
)

TOP_K = 5
MIN_SKILL_HITS = 1

WORD_SPLIT = re.compile(r"[^\w+#.]+")

def to_float(x, default=None):
    try:
        if x is None or str(x).strip()=="":
            return default
        return float(str(x).replace(",", "").strip())
    except:
        return default

def to_int(x, default=None):
    f = to_float(x, default)
    return int(f) if f is not None else default

def parse_list(s):
    if s is None: return []
    parts = re.split(r"[,\|;/]+", str(s))
    return [p.strip().lower() for p in parts if p.strip()]

def haversine_km(lat1, lon1, lat2, lon2):
    for v in (lat1, lon1, lat2, lon2):
        if v is None: return None
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2
         + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

_CURRENCY_MAP = {"$":"USD","usd":"USD","€":"EUR","eur":"EUR","£":"GBP","gbp":"GBP"}

def _parse_years_range(s: str):
    if not s: return (None, None)
    s = str(s).lower().strip()
    # Handles "0 to 12 Years", "3-5 years", "5+ years"
    m = re.findall(r"\d+\.?\d*", s)
    if not m: 
        return (None, None)
    nums = [float(x) for x in m]
    if "to" in s or "-" in s:
        lo = nums[0]
        hi = nums[1] if len(nums) > 1 else None
    elif "+" in s:
        lo = nums[0]; hi = None
    else:
        lo = nums[0]; hi = nums[0]
    return (lo, hi)

def _parse_salary_range(s: str):
    # "$61K-$104K", "€50,000–€70,000", "65000-90000", "Negotiable"
    if not s: return (None, None, "")
    raw = str(s).strip()
    # currency
    cur = ""
    for sym, code in _CURRENCY_MAP.items():
        if sym in raw.lower() or raw.startswith(sym) or raw.lower().endswith(sym):
            cur = code
            break
    # strip currency symbols and letters
    clean = re.sub(r"[^\d\.\-\–\,Kk]", "", raw)
    # expand K/k
    clean = re.sub(r"(\d+(?:\.\d+)?)\s*[Kk]", lambda m: str(float(m.group(1))*1000), clean)
    # replace en-dash
    clean = clean.replace("–", "-")
    parts = re.split(r"\s*-\s*", clean)
    try:
        lo = float(parts[0].replace(",", "")) if parts and parts[0] else None
        hi = float(parts[1].replace(",", "")) if len(parts) > 1 else None
    except:
        lo, hi = (None, None)
    return (lo, hi, cur)

def _bucket_company_size(val: str):
    # Turn "84525" into "1000+"
    try:
        n = int(str(val).replace(",", "").strip())
    except:
        return ""
    if n < 1: return "1-50"
    if n <= 50: return "1-50"
    if n <= 200: return "51-200"
    if n <= 1000: return "201-1000"
    return "1000+"

def _safe_json_or_text(s: str):
    # Some Kaggle fields look like JSON strings; try to load, else return as text
    if not s: return ""
    txt = str(s).strip()
    if txt.startswith("{") or txt.startswith("["):
        try:
            obj = json.loads(txt)
            return json.dumps(obj, ensure_ascii=False)  # normalized string
        except:
            return txt
    return txt

def normalize_kaggle_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase headers
    df.columns = [c.strip().lower() for c in df.columns]

    # Experience -> min/max years
    if "experience" in df.columns:
        mins, maxs = [], []
        for x in df["experience"]:
            lo, hi = _parse_years_range(x)
            mins.append(lo)
            maxs.append(hi)
        df["min years exp"] = mins
        df["max years exp"] = maxs

    # Salary Range -> salary min/max/currency
    if "salary range" in df.columns:
        lo_l, hi_l, cur_l = [], [], []
        for x in df["salary range"]:
            lo, hi, cur = _parse_salary_range(x)
            lo_l.append(lo); hi_l.append(hi); cur_l.append(cur)
        df["salary min"] = lo_l
        df["salary max"] = hi_l
        df["salary currency"] = cur_l

    # Company Size -> buckets
    if "company size" in df.columns:
        df["company size"] = df["company size"].apply(_bucket_company_size)

    # Tags from Preference / Job Portal
    tags = []
    for i in range(len(df)):
        row_tags = []
        pref = str(df.at[i, "preference"]).strip().lower() if "preference" in df.columns else ""
        if pref:
            # record as a tag (do NOT use to filter by gender)
            row_tags.append(f"preference-{pref.replace(' ', '-')}")
        jp = str(df.at[i, "job portal"]).strip().lower() if "job portal" in df.columns else ""
        if jp:
            row_tags.append(f"portal-{jp.replace(' ', '-')}")
        df.at[i, "tags"] = "|".join(row_tags) if row_tags else ""

    # Flatten JSON-ish long fields to enrich search blob
    for cname in ("benefits","responsibilities","company profile"):
        if cname in df.columns:
            df[cname] = df[cname].apply(_safe_json_or_text)

    # Build/extend searchable fields you already use
    # - job description: append responsibilities + benefits + company profile
    if "job description" in df.columns:
        jd = df["job description"].astype(str)
        resp = df["responsibilities"].astype(str) if "responsibilities" in df.columns else ""
        ben = df["benefits"].astype(str) if "benefits" in df.columns else ""
        prof = df["company profile"].astype(str) if "company profile" in df.columns else ""
        df["job description"] = (jd.fillna("") + " " + resp.fillna("") + " " + ben.fillna("") + " " + prof.fillna("")).str.strip()

    # Skills: ensure plain text list (your matcher does substring hits)
    if "skills" in df.columns:
        df["skills"] = df["skills"].astype(str)

    # Latitude/Longitude already present as strings—your code converts to float later
    return df

def main(jobs_csv: str, users_json: str):
    with open(users_json) as f:
        users = json.load(f)["users"]

    # 1) read CSV
    df = pd.read_csv(jobs_csv, dtype=str, keep_default_na=False, engine="c")
    df = normalize_kaggle_columns(df)
    if "country" in df.columns:
        df = df[df["country"].str.lower().str.contains(r"\b(?:united states|usa|u\.s\.a?\.?)\b", regex=True, na=False)]
        print(f"[INFO] Filtered to United States only → {len(df)} rows remaining.")

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

    # optional columns if existent
    min_exp  = col("min years exp").apply(to_int)
    max_exp  = col("max years exp").apply(to_int)
    sal_min  = col("salary min").apply(to_float)
    sal_max  = col("salary max").apply(to_float)
    sal_cur  = col("salary currency").str.upper()
    csize    = col("company size").str.strip().str.lower()
    lat_s    = col("latitude").apply(to_float)
    lon_s    = col("longitude").apply(to_float)
    tags_s   = col("tags").apply(parse_list)

    for u in users:
        name = u.get("name", u.get("user_id","(user)"))
        desired_roles = [s.strip().lower() for s in u.get("desired_roles", []) if s]
        skills = [s.strip().lower() for s in u.get("skills", []) if s]
        loc_keywords = [s.strip().lower() for s in u.get("location_keywords", []) if s]
        remote_ok = bool(u.get("remote_ok", True))

        yexp = u.get("years_experience", None)

        sal_pref = u.get("desired_salary", {}) or {}
        sal_pref_min = sal_pref.get("min", None)
        sal_pref_max = sal_pref.get("max", None)
        sal_pref_cur = (sal_pref.get("currency") or "").upper()

        wtypes_pref = [s.strip().lower() for s in u.get("work_type_pref", []) if s]
        csize_pref  = [s.strip().lower() for s in u.get("company_size_pref", []) if s]
        ctags_pref  = [s.strip().lower() for s in u.get("company_tags_pref", []) if s]

        geo_pref = u.get("geo_pref") or {}
        geo_lat  = geo_pref.get("lat")
        geo_lon  = geo_pref.get("lon")
        geo_rad  = geo_pref.get("radius_km")

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
            
        # Experience
        if yexp is not None and ("min years exp" in df.columns or "max years exp" in df.columns):
            exp_mask = pd.Series(True, index=df.index)
            if "min years exp" in df.columns:
                exp_mask &= (min_exp.fillna(0) <= yexp)
            if "max years exp" in df.columns:
                exp_mask &= (max_exp.fillna(10**9) >= yexp)
        else:
            exp_mask = pd.Series(True, index=df.index)

        # Salary overlap (+ currency, if provided)
        if (sal_pref_min is not None or sal_pref_max is not None) and ("salary min" in df.columns or "salary max" in df.columns):
            cur_ok = (sal_cur.eq(sal_pref_cur)) | (sal_cur.eq("")) | (sal_pref_cur == "")
            job_min = sal_min.fillna(-10**12)
            job_max = sal_max.fillna( 10**12)
            pref_min = sal_pref_min if sal_pref_min is not None else -10**12
            pref_max = sal_pref_max if sal_pref_max is not None else  10**12
            sal_mask = (job_max >= pref_min) & (job_min <= pref_max) & cur_ok
        else:
            sal_mask = pd.Series(True, index=df.index)

        # Work type
        if wtypes_pref and "work type" in df.columns:
            wtype_mask2 = pd.Series(False, index=df.index)
            for wt in wtypes_pref:
                wtype_mask2 |= wtype_l.str.contains(re.escape(wt), na=False)
        else:
            wtype_mask2 = pd.Series(True, index=df.index)

        # Company size
        if csize_pref and "company size" in df.columns:
            csize_mask = csize.isin(csize_pref)
        else:
            csize_mask = pd.Series(True, index=df.index)

        # Company tags
        if ctags_pref and "tags" in df.columns:
            ctags_mask = pd.Series(False, index=df.index)
            ctags_mask |= tags_s.apply(lambda lst: any(tag in lst for tag in ctags_pref))
        else:
            ctags_mask = pd.Series(True, index=df.index)

        # Geo radius
        if geo_lat is not None and geo_lon is not None and geo_rad and ("latitude" in df.columns and "longitude" in df.columns):
            dists = pd.Series(
                [haversine_km(geo_lat, geo_lon, lat_s.iloc[i], lon_s.iloc[i]) for i in range(len(df))],
                index=df.index
            )
            geo_mask = dists.notna() & (dists <= geo_rad)
        else:
            geo_mask = pd.Series(True, index=df.index)


        # candidate rows are any with role OR location hit
        hard_mask = exp_mask & sal_mask & wtype_mask2 & csize_mask
        cand_mask = hard_mask & (role_mask | loc_mask)
        cand_idx = df.index[cand_mask]

        # backoff 1: role-only
        if len(cand_idx) == 0:
            cand_mask = hard_mask & role_mask
            cand_idx = df.index[cand_mask]

        # backoff 2: location-only
        if len(cand_idx) == 0:
            cand_mask = hard_mask & loc_mask
            cand_idx = df.index[cand_mask]

        # backoff 3: just hard filters
        if len(cand_idx) == 0:
            cand_idx = df.index[hard_mask]

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

            exp_hit = 1 if exp_mask.loc[idx] else 0
            sal_hit = 1 if sal_mask.loc[idx] else 0
            wt_hit  = 1 if wtype_mask2.loc[idx] else 0
            cs_hit  = 1 if csize_mask.loc[idx] else 0
            tag_hit = 1 if ctags_mask.loc[idx] else 0
            geo_hit = 1 if geo_mask.loc[idx] else 0

            score = (
                WEIGHTS["skill"] * skill_hits +
                WEIGHTS["role"]  * role_hit +
                WEIGHTS["location"] * loc_hit +
                WEIGHTS["exp"]   * exp_mask.loc[idx] +
                WEIGHTS["salary"]* sal_mask.loc[idx] +
                WEIGHTS["wtype"] * wtype_mask2.loc[idx] +
                WEIGHTS["csize"] * csize_mask.loc[idx] +
                WEIGHTS["tags"]  * ctags_mask.loc[idx] +
                WEIGHTS["geo"]   * geo_mask.loc[idx]
            )

            if skill_hits < MIN_SKILL_HITS:
                continue

            scores.append((
                score, idx, skill_hits, role_hit, loc_hit,
                exp_hit, sal_hit, wt_hit, cs_hit, tag_hit, geo_hit
            ))

        if not scores and MIN_SKILL_HITS > 0:
            for idx in cand_idx:
                jb_blob = blob_series.loc[idx]
                text_l = jb_blob
                # re-evaluate with no skill threshold
                hits = 0
                if skills:
                    seen = set()
                    for s in skills:
                        if s and s not in seen and s in text_l:
                            hits += 1
                            seen.add(s)
                skill_hits = hits
                role_hit = 1 if role_mask.loc[idx] else 0
                loc_hit  = 1 if loc_mask.loc[idx]  else 0
                exp_hit  = 1 if exp_mask.loc[idx]  else 0
                sal_hit  = 1 if sal_mask.loc[idx]  else 0
                wt_hit   = 1 if wtype_mask2.loc[idx] else 0
                cs_hit   = 1 if csize_mask.loc[idx] else 0
                tag_hit  = 1 if ctags_mask.loc[idx] else 0
                geo_hit  = 1 if geo_mask.loc[idx] else 0

                score = (
                    WEIGHTS["skill"] * skill_hits +
                    WEIGHTS["role"]  * role_hit +
                    WEIGHTS["location"] * loc_hit +
                    WEIGHTS["exp"]   * exp_hit +
                    WEIGHTS["salary"]* sal_hit +
                    WEIGHTS["wtype"] * wt_hit +
                    WEIGHTS["csize"] * cs_hit +
                    WEIGHTS["tags"]  * tag_hit +
                    WEIGHTS["geo"]   * geo_hit
                )
                scores.append((score, idx, skill_hits, role_hit, loc_hit, exp_hit, sal_hit, wt_hit, cs_hit, tag_hit, geo_hit))
                
        scores.sort(key=lambda t: (t[0], t[2], t[3], t[4]), reverse=True)
        top = scores[:TOP_K]

        print("="*80)
        print(f"Top {len(top)} matches for: {name}")
        print("="*80)

        for rank, (score, idx, s_hits, r_hit, l_hit, e_hit, sa_hit, wt2, cs, tg, gh) in enumerate(top, 1):
            jt = (title.loc[idx] if isinstance(title, pd.Series) else "") or ""
            rl = (role.loc[idx] if isinstance(role, pd.Series) else "") or ""
            lc = (loc.loc[idx] if isinstance(loc, pd.Series) else "") or ""
            cn = (country.loc[idx] if isinstance(country, pd.Series) else "") or ""
            wt = (wtype.loc[idx] if isinstance(wtype, pd.Series) else "") or ""
            jid = df.loc[idx, "job id"] if "job id" in df.columns else "(n/a)"
            # extras if present
            exp_min = min_exp.loc[idx] if "min years exp" in df.columns else None
            exp_max = max_exp.loc[idx] if "max years exp" in df.columns else None
            smn = sal_min.loc[idx] if "salary min" in df.columns else None
            smx = sal_max.loc[idx] if "salary max" in df.columns else None
            scur= sal_cur.loc[idx] if "salary currency" in df.columns else ""

            print(f"{rank:>2}. {jt or '(no title)'}")
            print(f"    Job Id: {jid}")
            print(f"    Role: {rl} | Location: {lc}, {cn} | Work Type: {wt}")
            if exp_min is not None or exp_max is not None:
                print(f"    Experience: {exp_min if exp_min is not None else '?'}–{exp_max if exp_max is not None else '?'} yrs")
            if smn is not None or smx is not None or scur:
                print(f"    Salary: {smn if smn is not None else '?'}–{smx if smx is not None else '?'} {scur or ''}".strip())
            if "company size" in df.columns:
                print(f"    Company Size: {csize.loc[idx]}")
            if "tags" in df.columns:
                print(f"    Tags: {', '.join(tags_s.loc[idx])}")

            # --- Extra Kaggle fields (print only if they exist) ---
            if "job posting date" in df.columns:
                print(f"    Posted: {df.loc[idx, 'job posting date']}")

            if "job portal" in df.columns:
                print(f"    Source: {df.loc[idx, 'job portal']}")

            if "company" in df.columns:
                print(f"    Company: {df.loc[idx, 'company']}")

            if "contact person" in df.columns or "contact" in df.columns:
                cp = df.loc[idx, "contact person"] if "contact person" in df.columns else ""
                ct = df.loc[idx, "contact"] if "contact" in df.columns else ""
                if cp or ct:
                    print(f"    Recruiter: {cp} {ct}".strip())


            print(f"    Score: {score:.2f}  "
                  f"(skills:{s_hits} role:{bool(r_hit)} loc/remote:{bool(l_hit)} "
                  f"exp:{bool(e_hit)} sal:{bool(sa_hit)} wtype:{bool(wt2)} "
                  f"csize:{bool(cs)} tags:{bool(tg)} geo:{bool(gh)})")
            print("-"*80)
        print()

if __name__ == "__main__":
    JOBS_CSV = "./job_descriptions.csv"
    USERS_JSON = "./user_profiles.json"
    main(JOBS_CSV, USERS_JSON)