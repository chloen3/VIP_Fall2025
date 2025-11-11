export default function JobCard({
    title, 
    company, 
    location,
    qualifications, 
    workType,
    salary, 
    skills = [], 
    benefits = [], 
    postedAt, 
    responsibilities
}) {
    return (
      <article className="ui-card">
      <div className="flex gap-4">
        <div className="mt-1 h-10 w-10 flex-shrink-0 rounded-md bg-slate-200" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-2">
            <h3 className="font-semibold leading-tight truncate">{title}</h3>
            <span className="text-sm text-slate-600 truncate">@ {company}</span>
          </div>

          <div className="mt-1 text-sm text-slate-700">
            {location}
            {salary ? <> • {salary}</> : null}
            {workType ? <> • {workType}</> : null}
            {qualifications ? <> • {qualifications}</> : null}
          </div>

          {skills.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {skills.slice(0, 6).map((t, i) => (
                <span
                  key={i}
                  className="rounded-full border border-[#d6e7c5] bg-[#eff7e7] px-2 py-0.5 text-xs text-slate-700"
                >
                  {t}
                </span>
              ))}
            </div>
          )}

          {responsibilities && (
            <p className="mt-2 line-clamp-2 text-sm text-slate-700">
              {responsibilities}
            </p>
          )}

          {benefits.length > 0 && (
            <div className="mt-2 text-xs text-slate-600 truncate">
              Benefits: {benefits.slice(0, 4).join(" • ")}
            </div>
          )}

          {postedAt && <div className="mt-3 text-xs text-slate-500">{postedAt}</div>}
        </div>
      </div>
    </article>
  );
}