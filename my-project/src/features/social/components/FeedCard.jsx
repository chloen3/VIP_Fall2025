export default function FeedCard({ handle, text, right }) {
  return (
    <article className="ui-card">
      <div className="flex gap-4">
        <div className="mt-1 h-8 w-8 flex-shrink-0 rounded-full bg-slate-200" />
        <div className="min-w-0 flex-1">
          <div className="mb-1 font-semibold">{handle}</div>
          <p className="text-sm text-slate-700">{text}</p>
        </div>
        {right ? <div className="flex-shrink-0">{right}</div> : null}
      </div>
    </article>
  );
}