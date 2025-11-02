import json, re
import pandas as pd
from dotenv import load_dotenv

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

# ---------------- Configuration ----------------
TOP_K = 5
MIN_SKILL_HITS = 0
UPS_SKILLS_PER_JOB = 6          # how many suggestions to request per job
UPS_TOP_JOBS_FOR_SUGGEST = 5    # only ask OpenAI for the top N jobs to save tokens
OPENAI_MODEL = "gpt-4o-mini"

WORD_SPLIT = re.compile(r"[^\w+#.]+")
SKILL_SPLIT = re.compile(r"[,\|/;•\n]+")

# ---------------- OpenAI client ----------------
try:
    # OpenAI SDK v1
    from openai import OpenAI
except ImportError:
    OpenAI = None

def _load_openai_client():
    """
    Loads .env and returns an OpenAI client or None.
    Prefers API_KEY, falls back to OPENAI_API_KEY.
    """
    load_dotenv()  # loads .env if present
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        if OpenAI is None:
            print("[llm] OpenAI SDK not installed. Skipping suggestions.")
        else:
            print("[llm] No API key found in .env (API_KEY or OPENAI_API_KEY). Skipping suggestions.")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"[llm] Failed to init OpenAI client: {e}")
        return None

_client = _load_openai_client()

# ---------------- Helpers ----------------
def to_list_lower(x):
    if not x:
        return []
    if isinstance(x, list):
        return [str(s).strip().lower() for s in x]
    return [str(x).strip().lower()]

def tokenize(text: str):
    if pd.isna(text) or text is None:
        return []
    return [t for t in WORD_SPLIT.split(str(text).lower()) if t]

def count_skill_hits(text: str, skills):
    text_l = text.lower()
    hits = 0
    seen = set()
    for s in skills:
        if s and s not in seen and s in text_l:
            hits += 1
            seen.add(s)
    return hits

def extract_job_skills(text: str) -> list[str]:
    """Pull a clean list from a 'skills' field or a JD blob."""
    if not text:
        return []
    parts = [p.strip() for p in SKILL_SPLIT.split(str(text)) if p.strip()]
    if not parts:
        parts = tokenize(text)
    seen, out = set(), []
    for p in parts:
        p_l = p.lower()
        if len(p_l) < 2:
            continue
        if p_l in seen:
            continue
        seen.add(p_l)
        out.append(p)
    return out[:50]

# ---------------- LLM advice (batch) ----------------
def advise_for_jobs(jobs: list[dict], user_skills: list[str], *, model=OPENAI_MODEL) -> dict:
    """
    jobs: list of {"job_id": str, "title": str, "job_skills": [str]}
    user_skills: ["python", "excel", ...]
    Returns: {job_id: "Advice sentence ...", ...}
    """
    if _client is None:
        return {}

    sys = (
        "You are a concise career coach. For each job, write 1–2 sentences of concrete, "
        "actionable upskilling advice that would materially improve the candidate's fit. "
        "Reference 2–4 specific skills/experiences to add or deepen. Do not repeat skills the "
        "candidate already has unless suggesting an advanced or clinically-required credential. "
        "No fluff."
    )
    user_payload = {
        "candidate_current_skills": user_skills,
        "jobs": [{"job_id": j["job_id"], "title": j["title"], "job_skills": j.get("job_skills", [])}
                 for j in jobs]
    }
    user_msg = (
        "Return ONLY JSON of the form:\n"
        "{\n"
        '  "advice": [\n'
        '    {"job_id": "...", "title": "...", "sentence": "one or two sentences"},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        f"DATA:\n{json.dumps(user_payload)[:9000]}"
    )

    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        print("[llm] API call OK — received bytes:", len(content))  # <-- debug success print
    except Exception as e:
        print("[llm] API call FAILED:", e)
        return {}

    try:
        data = json.loads(content)
        items = data.get("advice", [])
        out = {}
        for it in items:
            jid = str(it.get("job_id", "")).strip()
            sent = str(it.get("sentence", "")).strip()
            if jid and sent:
                out[jid] = sent
        return out
    except Exception as e:
        print("[llm] Parse error:", e)
        return {}

# ---------------- Main ----------------
def main(jobs_csv: str, users_json: str):
    with open(users_json) as f:
        users = json.load(f)["users"]

    # only take the first user for quick testing
    users = users[:1]

    # 1) read CSV
    df = pd.read_csv(jobs_csv, dtype=str, keep_default_na=False, engine="c")
    df = normalize_kaggle_columns(df)
    if "country" in df.columns:
        df = df[df["country"].str.lower().str.contains(r"\b(?:united states|usa|u\.s\.a?\.?)\b", regex=True, na=False)]
        print(f"[INFO] Filtered to United States only → {len(df)} rows remaining.")

    df.columns = [c.strip().lower() for c in df.columns]

    def col(name):
        return df[name] if name in df.columns else pd.Series([""] * len(df), index=df.index)

    # column references
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

        # build blobs ONLY for the candidates (for your existing scoring)
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

        for rank, (score, idx, s_hits, r_hit, l_hit) in enumerate(top, 1):
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
            print(f"    Score: {score}  (skills:{s_hits}  role:{bool(r_hit)}  loc/remote:{bool(l_hit)})")
            print("-"*80)

        # ---------------- Upskilling advice (only for top N) ----------------
        if upskill_payloads:
            jobs_for_advice = []
            for (rank, jid, jt, jdesc, idx) in upskill_payloads[:UPS_TOP_JOBS_FOR_SUGGEST]:
                raw_skills = jskills.loc[idx] if isinstance(jskills, pd.Series) else ""
                job_sk_list = extract_job_skills(raw_skills) or extract_job_skills(jdesc)
                jobs_for_advice.append({
                    "job_id": str(jid),
                    "title": jt,
                    "job_skills": job_sk_list,
                })

            user_current_skills = skills

            print("\nUpskilling suggestions to become an even stronger candidate:")
            advice_map = advise_for_jobs(jobs_for_advice, user_current_skills, model=OPENAI_MODEL)

            for (rank, jid, jt, _jdesc, _idx) in upskill_payloads[:UPS_TOP_JOBS_FOR_SUGGEST]:
                sent = advice_map.get(str(jid), "")
                if sent:
                    print(f"  [{rank}] {jt} (Job Id: {jid})")
                    print(f"     {sent}")
                else:
                    print(f"  [{rank}] {jt} (Job Id: {jid}) — (no advice returned)")
            print()

if __name__ == "__main__":
    JOBS_CSV = "./job_descriptions.csv"
    USERS_JSON = "./user_profiles.json"
    main(JOBS_CSV, USERS_JSON)
