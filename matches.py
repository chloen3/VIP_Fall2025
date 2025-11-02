import os, json, re, time
import pandas as pd
from dotenv import load_dotenv

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
            loc_mask |= (
                loc_l.str.contains("remote", regex=False, na=False) |
                country_l.str.contains("remote", regex=False, na=False) |
                wtype_l.str.contains("remote", regex=False, na=False) |
                desc_l.str.contains("remote", regex=False, na=False)
            )

        # candidate rows are any with role OR location hit
        cand_idx = df.index[role_mask | loc_mask]
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

            score = skill_hits + role_hit + loc_hit
            if skill_hits < MIN_SKILL_HITS:
                continue

            scores.append((score, idx, skill_hits, role_hit, loc_hit))

        scores.sort(key=lambda t: (t[0], t[2], t[3], t[4]), reverse=True)
        top = scores[:TOP_K]

        print("="*80)
        print(f"Top {len(top)} matches for: {name}")
        print("="*80)

        # Pre-collect text for advice calls (top N jobs only)
        upskill_payloads = []
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

            if rank <= UPS_TOP_JOBS_FOR_SUGGEST:
                jdesc = " ".join([
                    str(title.loc[idx] or ""),
                    str(role.loc[idx] or ""),
                    str(desc.loc[idx] or ""),
                    str(jskills.loc[idx] or "")
                ])
                upskill_payloads.append((rank, jid, jt, jdesc, idx))

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
