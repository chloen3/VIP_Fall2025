import { useState } from "react";
import Tabs from "./components/Tabs.jsx";
import FeedList from "./components/FeedList.jsx";
import JobsList from "./components/JobList.jsx";
//import ResourcesList from "./components/ResourcesList.jsx";

export default function SocialPanel() {
  const [active, setActive] = useState("feed");
  const [jobQuery, setJobQuery] = useState("");

  return (
    <section>
      <Tabs active={active} onChange={setActive} />
      {active === "jobs" && (
        <div className="mb-3">
          <label className="sr-only" htmlFor="job-search">Search jobs</label>
          <input
            id="job-search"
            type="text"
            value={jobQuery}
            onChange={(e) => setJobQuery(e.target.value)}
            placeholder="Search title, company, location, skills…"
            className="w-full rounded-xl border border-[#d6e7c5] bg-white/70 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[#5aa56a]"
          />
        </div>
      )}
      {active === "feed" && <FeedList />}
      {active === "jobs" && <JobsList query={jobQuery} pageSize={5}/>}
      {active === "resources" && <ResourcesList />}
    </section>
  );
}