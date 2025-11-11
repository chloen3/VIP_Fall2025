export default function CommunitiesCard() {
  return (
    <section className="rounded-2xl border border-slate-200 bg-[#e6f1dc] p-4">
      <h4 className="mb-3 text-base font-semibold text-[#416a4a]">Join Communities</h4>
      <div className="space-y-3">
        {[1,2,3].map(n => (
          <div key={n} className="flex items-center justify-between rounded-xl bg-white p-3 shadow-sm ring-1 ring-slate-200">
            <div className="text-sm font-medium">Community {n}</div>
            <div className="flex -space-x-1">
              
            </div>
          </div>
        ))}
      </div>
      <a href="#" className="mt-3 inline-block text-sm text-[#416a4a] underline">view more →</a>
    </section>
  );
}