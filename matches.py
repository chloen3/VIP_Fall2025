import os, json, re, math, time
import pandas as pd
from dotenv import load_dotenv


# Weights & Configuration
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
MIN_SKILL_HITS = 0
UPS_SKILLS_PER_JOB = 6
UPS_TOP_JOBS_FOR_SUGGEST = 5
OPENAI_MODEL = "gpt-4o-mini"

WORD_SPLIT = re.compile(r"[^\w+#.]+")
SKILL_SPLIT = re.compile(r"[,\|/;•\n]+")

# OpenAI client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import numpy as np
except ImportError:
    np = None

def _load_openai_client():
    """Loads .env and returns an OpenAI client or None. Prefers API_KEY, falls back to OPENAI_API_KEY."""
    load_dotenv()
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

EMB_MODEL = "text-embedding-3-small"  # cheap & strong; or -large for best quality

def _embed_texts(client, texts, model=EMB_MODEL, batch=96):
    if client is None or np is None:
        return None
    vecs = []
    for i in range(0, len(texts), batch):
        sl = texts[i:i+batch]
        r = client.embeddings.create(model=model, input=sl)
        vecs.extend([d.embedding for d in r.data])
    return np.array(vecs, dtype="float32")

def _cosine(a, b):
    # a: (d,), b: (n,d)
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b @ a

def build_job_embeddings(df, title, role, desc, jskills):
    """Create one text blob per job, then embed (cache to speed up)."""
    if _client is None or np is None:
        return None
    blobs = []
    idxs = []
    for i in df.index:
        t = str(title.loc[i] or "")
        r = str(role.loc[i] or "")
        d = str(desc.loc[i] or "")
        s = str(jskills.loc[i] or "")
        blobs.append((t + " | " + r + " | " + s + " | " + d)[:3000])  # clip
        idxs.append(i)
    embs = _embed_texts(_client, blobs)
    return (np.array(idxs), embs)

def embed_user_profile(skills, desired_roles=None):
    if _client is None or np is None:
        return None
    text = "Skills: " + ", ".join(skills)
    if desired_roles:
        text += " | Desired roles: " + ", ".join(desired_roles)
    v = _embed_texts(_client, [text])
    return v[0] if v is not None else None

def semantic_explain_alignment(
    client,
    user_skills: list[str],
    job_text: str,
    *,
    emb_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    topk_per_skill=2,
    max_terms=120,
    include_themes=True
):
    if client is None or np is None or not user_skills or not job_text:
        return {"bridges": [], "themes": ""}

    # --- Extract candidate terms from job text ---
    stop = set("""
        a an the and or for of to in on with by as from at into over under between across
        is are was were be being been have has had do does did can could should would will may might
        this that these those it its their our your his her them they you we i
    """.split())

    toks = [t.strip().lower() for t in re.split(r"[^\w#+./-]+", job_text) if t.strip()]
    toks = [t for t in toks if len(t) >= 2 and t not in stop]
    pref = [t for t in toks if any(c in t for c in "#+./-") or any(ch.isdigit() for ch in t) or len(t) > 5]

    bigrams = [
        f"{toks[i]} {toks[i+1]}"
        for i in range(len(toks) - 1)
        if toks[i] not in stop and toks[i+1] not in stop
    ]

    jd_terms = []
    seen = set()
    for t in pref + bigrams + toks:
        if t not in seen:
            seen.add(t)
            jd_terms.append(t)
        if len(jd_terms) >= max_terms:
            break

    if not jd_terms:
        return {"bridges": [], "themes": ""}

    # --- Compute embeddings for skills and JD terms ---
    skills = [s.strip().lower() for s in user_skills if s and str(s).strip()]
    skill_vecs = _embed_texts(client, skills, model=emb_model)
    term_vecs = _embed_texts(client, jd_terms, model=emb_model)
    if skill_vecs is None or term_vecs is None:
        return {"bridges": [], "themes": ""}
    term_norm = term_vecs / (np.linalg.norm(term_vecs, axis=1, keepdims=True) + 1e-9)

    # --- Build bridges: per-skill top semantic matches ---
    bridges = []
    for s, sv in zip(skills, skill_vecs):
        v = sv / (np.linalg.norm(sv) + 1e-9)
        sims = term_norm @ v
        order = np.argsort(-sims)
        pairs = [(jd_terms[j], float(max(0.0, sims[j]))) for j in order[:topk_per_skill] if sims[j] > 0]
        if pairs:
            bridges.append({"skill": s, "matches": pairs})

    # --- Optional: summarize themes with a short LLM call ---
    themes = ""
    if include_themes and bridges:
        try:
            small = [{"skill": b["skill"], "matches": [m[0] for m in b["matches"][:2]]}
                     for b in bridges[:6]]
            sys = (
                "You explain conceptual alignments between a candidate's skills and a job posting. "
                "Write 1–2 compact lines that highlight themes like "
                "'Data viz → Tableau/Looker' or 'Project Mgmt → Scrum/Jira'."
            )
            user = {"bridges": small}
            r = client.chat.completions.create(
                model=llm_model,
                temperature=0.2,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
            )
            themes = r.choices[0].message.content.strip()
        except Exception as e:
            print("[llm] summarize themes failed:", e)

    return {"bridges": bridges, "themes": themes}

# Helpers
def to_float(x, default=None):
    try:
        if x is None or str(x).strip() == "":
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
    cur = ""
    for sym, code in _CURRENCY_MAP.items():
        if sym in raw.lower() or raw.startswith(sym) or raw.lower().endswith(sym):
            cur = code
            break
    clean = re.sub(r"[^\d\.\-\–\,Kk]", "", raw)
    clean = re.sub(r"(\d+(?:\.\d+)?)\s*[Kk]", lambda m: str(float(m.group(1))*1000), clean)
    clean = clean.replace("–", "-")
    parts = re.split(r"\s*-\s*", clean)
    try:
        lo = float(parts[0].replace(",", "")) if parts and parts[0] else None
        hi = float(parts[1].replace(",", "")) if len(parts) > 1 else None
    except:
        lo, hi = (None, None)
    return (lo, hi, cur)

def _bucket_company_size(val: str):
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
    if not s: return ""
    txt = str(s).strip()
    if txt.startswith("{") or txt.startswith("["):
        try:
            obj = json.loads(txt)
            return json.dumps(obj, ensure_ascii=False)
        except:
            return txt
    return txt

def normalize_kaggle_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]

    if "experience" in df.columns:
        mins, maxs = [], []
        for x in df["experience"]:
            lo, hi = _parse_years_range(x)
            mins.append(lo); maxs.append(hi)
        df["min years exp"] = mins
        df["max years exp"] = maxs

    if "salary range" in df.columns:
        lo_l, hi_l, cur_l = [], [], []
        for x in df["salary range"]:
            lo, hi, cur = _parse_salary_range(x)
            lo_l.append(lo); hi_l.append(hi); cur_l.append(cur)
        df["salary min"] = lo_l
        df["salary max"] = hi_l
        df["salary currency"] = cur_l

    if "company size" in df.columns:
        df["company size"] = df["company size"].apply(_bucket_company_size)

    tags = []
    for i in range(len(df)):
        row_tags = []
        pref = str(df.at[i, "preference"]).strip().lower() if "preference" in df.columns else ""
        if pref:
            row_tags.append(f"preference-{pref.replace(' ', '-')}")
        jp = str(df.at[i, "job portal"]).strip().lower() if "job portal" in df.columns else ""
        if jp:
            row_tags.append(f"portal-{jp.replace(' ', '-')}")
        df.at[i, "tags"] = "|".join(row_tags) if row_tags else ""

    for cname in ("benefits","responsibilities","company profile"):
        if cname in df.columns:
            df[cname] = df[cname].apply(_safe_json_or_text)

    if "job description" in df.columns:
        jd = df["job description"].astype(str)
        resp = df["responsibilities"].astype(str) if "responsibilities" in df.columns else ""
        ben = df["benefits"].astype(str) if "benefits" in df.columns else ""
        prof = df["company profile"].astype(str) if "company profile" in df.columns else ""
        df["job description"] = (jd.fillna("") + " " + resp.fillna("") + " " + ben.fillna("") + " " + prof.fillna("")).str.strip()

    if "skills" in df.columns:
        df["skills"] = df["skills"].astype(str)

    return df

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

# LLM advice
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
        '  \"advice\": [\n'
        '    {\"job_id\": \"...\", \"title\": \"...\", \"sentence\": \"one or two sentences\"},\n'
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
        print("[llm] API call OK — received bytes:", len(content))  # Debug success print
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

# Main
def main(jobs_csv: str, users_json: str):
    with open(users_json) as f:
        users = json.load(f)["users"]

    # Only the first user for quick test
    users = users[:1]

    # Load and normalize jobs
    df = pd.read_csv(jobs_csv, dtype=str, keep_default_na=False, na_values=[])
    df = normalize_kaggle_columns(df)
    df.columns = [c.strip().lower() for c in df.columns]

    def col(name):
        return df[name] if name in df.columns else pd.Series([""] * len(df), index=df.index)

    # Field references
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

    # Optional columns
    min_exp  = col("min years exp").fillna("").apply(to_int)
    max_exp  = col("max years exp").fillna("").apply(to_int)
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

        # Role mask
        if desired_roles:
            role_mask = pd.Series(False, index=df.index)
            for term in desired_roles:
                if term:
                    role_mask |= title_l.str.contains(re.escape(term), na=False)
                    role_mask |= role_l.str.contains(re.escape(term), na=False)
                    role_mask |= desc_l.str.contains(re.escape(term), na=False)
        else:
            role_mask = pd.Series(False, index=df.index)

        # Location mask
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

        if remote_ok:
            loc_mask |= (loc_l.str.contains("remote", regex=False, na=False) |
                         country_l.str.contains("remote", regex=False, na=False) |
                         wtype_l.str.contains("remote", regex=False, na=False) |
                         desc_l.str.contains("remote", regex=False, na=False))

        # Experience mask
        if yexp is not None and ("min years exp" in df.columns or "max years exp" in df.columns):
            exp_mask = pd.Series(True, index=df.index)
            if "min years exp" in df.columns:
                exp_mask &= (min_exp.fillna(0) <= yexp)
            if "max years exp" in df.columns:
                exp_mask &= (max_exp.fillna(10**9) >= yexp)
        else:
            exp_mask = pd.Series(True, index=df.index)

        # Salary mask
        if (sal_pref_min is not None or sal_pref_max is not None) and ("salary min" in df.columns or "salary max" in df.columns):
            cur_ok = (sal_cur.eq(sal_pref_cur)) | (sal_cur.eq("")) | (sal_pref_cur == "")
            job_min = sal_min.fillna(-10**12)
            job_max = sal_max.fillna( 10**12)
            pref_min = sal_pref_min if sal_pref_min is not None else -10**12
            pref_max = sal_pref_max if sal_pref_max is not None else  10**12
            sal_mask = (job_max >= pref_min) & (job_min <= pref_max) & cur_ok
        else:
            sal_mask = pd.Series(True, index=df.index)

        # Work type mask
        if wtypes_pref and "work type" in df.columns:
            wtype_mask2 = pd.Series(False, index=df.index)
            for wt in wtypes_pref:
                wtype_mask2 |= wtype_l.str.contains(re.escape(wt), na=False)
        else:
            wtype_mask2 = pd.Series(True, index=df.index)

        # Company size mask
        if csize_pref and "company size" in df.columns:
            csize_mask = csize.isin(csize_pref)
        else:
            csize_mask = pd.Series(True, index=df.index)

        # Company tags mask
        if ctags_pref and "tags" in df.columns:
            ctags_mask = pd.Series(False, index=df.index)
            ctags_mask |= tags_s.apply(lambda lst: any(tag in lst for tag in ctags_pref))
        else:
            ctags_mask = pd.Series(True, index=df.index)

        # Geo mask
        if geo_lat is not None and geo_lon is not None and geo_rad and ("latitude" in df.columns and "longitude" in df.columns):
            dists = pd.Series(
                [haversine_km(geo_lat, geo_lon, lat_s.iloc[i], lon_s.iloc[i]) for i in range(len(df))],
                index=df.index
            )
            geo_mask = dists.notna() & (dists <= geo_rad)
        else:
            geo_mask = pd.Series(True, index=df.index)

        # Candidate set with backoffs
        hard_mask = exp_mask & sal_mask & wtype_mask2 & csize_mask
        cand_mask = hard_mask & (role_mask | loc_mask)
        cand_idx = df.index[cand_mask]
        if len(cand_idx) == 0:
            cand_mask = hard_mask & role_mask
            cand_idx = df.index[cand_mask]
        if len(cand_idx) == 0:
            cand_mask = hard_mask & loc_mask
            cand_idx = df.index[cand_mask]
        if len(cand_idx) == 0:
            cand_idx = df.index[hard_mask]

        if len(cand_idx) == 0:
            print("="*80)
            print(f"Top 0 matches for: {name}")
            print("="*80 + "\n")
            continue

        # Prepare blobs for scoring
        title_c   = title_l.loc[cand_idx]
        role_c    = role_l.loc[cand_idx]
        desc_c    = desc_l.loc[cand_idx]
        loc_c     = loc_l.loc[cand_idx]
        country_c = country_l.loc[cand_idx]
        wtype_c   = wtype_l.loc[cand_idx]
        jskills_c = jskills_l.loc[cand_idx]

        blob_series = (title_c + " " + role_c + " " + desc_c + " " + jskills_c)
        loc_series  = (loc_c + " " + country_c + " " + wtype_c + " " + desc_c)

        # --- build embeddings once for candidate set ---
        job_emb_cache = build_job_embeddings(df.loc[cand_idx], title_c, role_c, desc_c, jskills_c)
        if job_emb_cache is not None:
            cand_order, cand_embs = job_emb_cache  # cand_order aligns to cand_idx
            user_vec = embed_user_profile(skills, desired_roles)
        else:
            cand_order, cand_embs, user_vec = None, None, None

        scores = []
        for idx in cand_idx:
            jb_blob = blob_series.loc[idx]

            # Base hits
            skill_hits = count_skill_hits(jb_blob, skills) if skills else 0
            role_hit = 1 if role_mask.loc[idx] else 0
            loc_hit  = 1 if loc_mask.loc[idx]  else 0
            exp_hit  = 1 if exp_mask.loc[idx]  else 0
            sal_hit  = 1 if sal_mask.loc[idx]  else 0
            wt_hit   = 1 if wtype_mask2.loc[idx] else 0
            cs_hit   = 1 if csize_mask.loc[idx] else 0
            tag_hit  = 1 if ctags_mask.loc[idx] else 0
            geo_hit  = 1 if geo_mask.loc[idx] else 0

            # --- Semantic similarity (embeddings) ---
            sem_sim = 0.0
            if (cand_embs is not None) and (user_vec is not None):
                try:
                    # find row position of idx in the candidate embedding matrix
                    pos = int(np.where(cand_order == idx)[0][0])
                    sem_sim = float(_cosine(user_vec, cand_embs[pos:pos+1]))
                    # ignore negative similarity values (dissimilar)
                    if sem_sim < 0.0:
                        sem_sim = 0.0
                except Exception:
                    sem_sim = 0.0

            # --- Final weighted score ---
            SEM_WEIGHT = 1.2
            score = (
                WEIGHTS["skill"] * skill_hits +
                WEIGHTS["role"]  * role_hit +
                WEIGHTS["location"] * loc_hit +
                WEIGHTS["exp"]   * exp_mask.loc[idx] +
                WEIGHTS["salary"]* sal_mask.loc[idx] +
                WEIGHTS["wtype"] * wtype_mask2.loc[idx] +
                WEIGHTS["csize"] * csize_mask.loc[idx] +
                WEIGHTS["tags"]  * ctags_mask.loc[idx] +
                WEIGHTS["geo"]   * geo_mask.loc[idx] +
                SEM_WEIGHT       * sem_sim
            )

            if skill_hits < MIN_SKILL_HITS:
                continue

            scores.append((
                score, idx, skill_hits, role_hit, loc_hit,
                exp_hit, sal_hit, wt_hit, cs_hit, tag_hit, geo_hit
            ))

        # Optional backoff remove skill threshold
        if not scores and MIN_SKILL_HITS > 0:
            for idx in cand_idx:
                jb_blob = blob_series.loc[idx]
                skill_hits = count_skill_hits(jb_blob, skills) if skills else 0
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

        upskill_payloads = []
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

            alignment = semantic_explain_alignment(
                _client,
                skills,
                " ".join([
                    str(title.loc[idx] or ""),
                    str(role.loc[idx] or ""),
                    str(jskills.loc[idx] or ""),
                    str(desc.loc[idx] or "")
                ]),
                emb_model=EMB_MODEL,
                llm_model=OPENAI_MODEL,
                include_themes=True
            )
            bridges = alignment["bridges"]
            themes  = alignment["themes"]

            if bridges:
                print("    Concept bridges:")
                for b in bridges[:3]:  # show top 3 skills
                    pairs = ", ".join([f"{term} ({sim:.2f})" for term, sim in b["matches"][:2]])
                    print(f"       - {b['skill']} → {pairs}")
            if themes:
                for line in themes.splitlines():
                    print(f"       {line.strip()}")

            print(f"    Score: {score:.2f}  "
                  f"(skills:{s_hits} role:{bool(r_hit)} loc/remote:{bool(l_hit)} "
                  f"exp:{bool(e_hit)} sal:{bool(sa_hit)} wtype:{bool(wt2)} "
                  f"csize:{bool(cs)} tags:{bool(tg)} geo:{bool(gh)})")
            print("-"*80)

            # Collect payload for LLM (top N only)
            if rank <= UPS_TOP_JOBS_FOR_SUGGEST:
                jdesc = " ".join([
                    str(title.loc[idx] or ""),
                    str(role.loc[idx] or ""),
                    str(desc.loc[idx] or ""),
                    str(jskills.loc[idx] or "")
                ])
                upskill_payloads.append((rank, jid, jt, jdesc, idx))

        # ---- Upskilling advice (top N) ----
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
    JOBS_CSV = "./job_descriptions_us.csv"
    USERS_JSON = "./user_profiles.json"
    main(JOBS_CSV, USERS_JSON)
