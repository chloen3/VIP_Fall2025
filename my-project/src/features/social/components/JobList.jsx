import { useMemo, useState } from "react";
import JobCard from "./JobCard.jsx"
import raw from "/Users/emmawarren/Desktop/VIP-UI/my-project/src/data/jobs.json"

const parseBenefits = (s) =>
  s ? s.replace(/^\{\s*'?|'\s*\}$/g, "").split(",").map(x => x.trim()).filter(Boolean) : [];

const parseSkills = (s) =>
  !s ? [] : (/[;,|]/.test(s) ? s.split(/[;,|]/).map(t => t.trim()) : [s.trim()]);

const mapJob = (j) => ({
  title: j["Job Title"],
  company: j["Company"],
  location: [j["location"], j["Country"]].filter(Boolean).join(", "),
  qualifications: j["Qualifications"],
  workType: j["Work Type"],
  salary: j["Salary Range"],
  skills: parseSkills(j["skills"]),
  benefits: parseBenefits(j["Benefits"]),
  postedAt: j["Job Posting Date"],
  responsibilities: j["Responsibilities"],
});

export default function JobsList({ query = "", pageSize = 4 }) {
  const [page, setPage] = useState(1);

  const jobs = useMemo(() => raw.map(mapJob), []);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return jobs;
    return jobs.filter(j =>
      [j.title, j.company, j.location, j.qualifications, j.workType, j.salary, ...(j.skills||[])]
        .join(" ")
        .toLowerCase()
        .includes(q)
    );
  }, [jobs, query]);

  const total = filtered.length;
  const maxPage = Math.max(1, Math.ceil(total / pageSize));
  const start = (page - 1) * pageSize;
  const visible = filtered.slice(start, start + pageSize);

  // reset to page 1 when query changes to avoid empty pages
  if (page > maxPage) setPage(1);

  if (!total) {
    return (
      <section className="rounded-[18px] border border-[#d6e7c5] bg-[#eff7e7] p-5 min-h-[220px] flex items-center justify-center">
        <p className="text-slate-500 text-sm">No jobs yet.</p>
      </section>
    );
  }

  return (
    <section className="rounded-[18px] border border-[#d6e7c5] bg-[#eff7e7] p-5">
      <div className="space-y-4 max-h-[520px] overflow-auto pr-1">
        {visible.map((j, i) => <JobCard key={`${j.company}-${j.title}-${start+i}`} {...j} />)}
      </div>

      {/* controls */}
      <div className="mt-4 flex items-center justify-between text-sm">
        <span className="text-slate-600">
          Showing {start + 1}–{Math.min(start + pageSize, total)} of {total}
        </span>
        <div className="flex gap-2">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            className="rounded-full border px-3 py-1 disabled:opacity-40"
          >
            ← Prev
          </button>
          <button
            onClick={() => setPage(p => Math.min(maxPage, p + 1))}
            className="rounded-full border px-3 py-1 disabled:opacity-40"
          >
            Next →
          </button>
        </div>
      </div>
    </section>
  );
}