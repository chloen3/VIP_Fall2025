export default function MentorsCard() {
  return (
    <section className="rounded-2xl border border-slate-200 bg-[#e6f1dc] p-4">
      <h4 className="mb-3 text-base font-semibold text-[#416a4a]">Reach out to Mentors</h4>
      <div className="flex items-center gap-3">
        {[0,1,2,3].map(i => (
          <div key={i} className="h-10 w-10 rounded-full bg-white shadow ring-1 ring-slate-200" />
        ))}
      </div>
      <a href="#" className="mt-3 inline-block text-sm text-[#416a4a] underline">view more →</a>
    </section>
  );
}