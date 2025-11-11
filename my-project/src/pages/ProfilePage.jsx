import data from "/Users/emmawarren/Desktop/VIP-UI/my-project/src/data/users.json"
import headshot from "/Users/emmawarren/Desktop/VIP-UI/my-project/src/assets/headshot.jpeg"
import { Link } from "react-router-dom"

export default function ProfilePage() {
    const u = data.users.find(x => x.user_id === "milspouse-001") ?? data.users[0];
    return (
        <main className="max-w-3xl mx-auto p-6 space-y-6">
            <div>
                <Link
                    to="/"
                    className="inline-flex items-center gap-2 rounded-full border border-slate-200 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
                >
                    Back
                    <span className="sr-only">to home</span>
                </Link>
            </div>

            <header className="flex items-center gap-4">
                {/* <div className="h-16 w-16 rounded-full bg-slate-200" /> */}
                <div>
                    <img
                        src={headshot}
                        alt="Avery's Profile Picture"
                        className="h-28 w-28 rounded-full object-cover object-center bg-slate-200"
                    />
                    <h1 className="text-xl font-semibold">{u.name}</h1>
                    <p className="text-slate-700">{u.headline}</p>
                </div>
            </header>

        <section className="ui-card">
            <h2 className="font-semibold mb-2">Overview</h2>
            <div className="text-sm text-slate-700 space-y-1">
            <p><span className="font-medium">Experience:</span> {u.years_experience} years</p>
            <p><span className="font-medium">Remote OK:</span> {u.remote_ok ? "Yes" : "No"}</p>
            <p><span className="font-medium">Desired Salary:</span> ${u.desired_salary.min.toLocaleString()}–${u.desired_salary.max.toLocaleString()} {u.desired_salary.currency}</p>
            <p><span className="font-medium">States:</span> {u.states.join(", ").toUpperCase()}</p>
            <p><span className="font-medium">Licenses:</span> {u.state_licenses.join(", ")}</p>
            </div>
        </section>

        <section className="ui-card">
            <h2 className="font-semibold mb-2">Education</h2>
            <ul className="list-disc pl-5 text-sm text-slate-700">
            {u.education.map((e,i)=> <li key={i}>{e}</li>)}
            </ul>
        </section>

        <section className="ui-card">
            <h2 className="font-semibold mb-2">Desired Roles</h2>
            <div className="flex flex-wrap gap-2">
            {u.desired_roles.map((r,i)=>(
                <span key={i} className="px-2 py-0.5 text-xs">{r}</span>
            ))}
            </div>
        </section>

        <section className="ui-card">
            <h2 className="font-semibold mb-2">Skills</h2>
            <div className="flex flex-wrap gap-2">
            {u.skills.map((s,i)=>(
                <span key={i} className="px-2 py-0.5 text-xs">{s}</span>
            ))}
            </div>
        </section>

        <section className="ui-card">
            <h2 className="font-semibold mb-2">Preferences</h2>
            <div className="grid sm:grid-cols-2 gap-3 text-sm text-slate-700">
            <div><div className="font-medium">Work Type</div>{u.work_type_pref.join(", ")}</div>
            <div><div className="font-medium">Company Size</div>{u.company_size_pref.join(", ")}</div>
            <div>
                <div className="font-medium">Company Tags</div>
                <div className="flex flex-wrap gap-2 mt-1">
                {u.company_tags_pref.map((t,i)=>(
                    <span key={i} className="px-2 py-0.5 text-xs">{t}</span>
                ))}
                </div>
            </div>
            <div>
                <div className="font-medium">Geo Pref</div>
                <div>lat {u.geo_pref.lat}, lon {u.geo_pref.lon} • {u.geo_pref.radius_km} km</div>
            </div>
            <div className="sm:col-span-2">
                <div className="font-medium">Notes</div>
                <div>{u.notes}</div>
            </div>
            </div>
        </section>
    </main>
  );
}