export default function Tabs({active, onChange}) {
  const btn = (key, label) => (
    <button
      role="tab"
      aria-selected={active === key}
      onClick={() => onChange(key)}
      className={[
        "ui-tab", 
        active === key ? "ui-tab--active" : "", 
        "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#5aa56a]", 
      ].join(" ")}
    >
        {label}
    </button>
  );
  
  return (
    <div role="tablist" aria-label="Feed tabs" className="mb-4 flex flex-wrap items-center gap-2">
      {btn("feed", "Feed")}
      {btn("jobs", "Jobs")}
      {btn("resources", "Resources")}
    </div>
  );
}