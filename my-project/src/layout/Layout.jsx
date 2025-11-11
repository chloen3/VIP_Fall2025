export default function Layout({ header, sidebar, children }) {
  return (
    <div className="min-h-screen bg-[#f6f7f5] text-slate-900">
      {header}
      <div className="mx-auto max-w-6xl px-4 py-6">
        <div className="grid grid-cols-12 gap-6">
          <aside className="col-span-12 md:col-span-3 space-y-6">
            {sidebar}
          </aside>
          <main className="col-span-12 md:col-span-9">
            {children}
          </main>
        </div>
      </div>
    </div>
  );
}