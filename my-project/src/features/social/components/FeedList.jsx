import FeedCard from "./FeedCard.jsx";

export default function FeedList() {
  return (
    <section className="rounded-[18px] border border-[#d6e7c5] bg-[#eff7e7] p-5 space-y-5">
      <FeedCard handle="@User1" text="He moonlight difficult engrossed an it sportsmen. Interested has all devonshire difficulty gay assistance joy. Unaffected at ye of compliment alteration to. Place voice no arise along to. Parlors waiting so against me no. Wishing calling are warrant settled was luckily. Express besides it present if at an opinion visitor." />
      <FeedCard
        handle="@User2"
        text="Do play they miss give so up. Words to up style of since world. We leaf to snug on no need. Way own uncommonly travelling now acceptance bed compliment solicitude. Dissimilar admiration so terminated no in contrasted it. Advantages entreaties mr he apartments do. Limits far yet turned highly repair parish talked six. Draw fond rank form nor the day eat.
"
        right={<div className="h-20 w-28 rounded-md bg-slate-200" />}
      />
      <FeedCard
        handle="@User3"
        text="Attachment apartments in delightful by motionless it no. And now she burst sir learn total. Hearing hearted shewing own ask. Solicitude uncommonly use her motionless not collecting age. The properly servants required mistaken outlived bed and. Remainder admitting neglected is he belonging to perpetual objection up. Has widen too you decay begin which asked equal any."
        right={<div className="h-24 w-36 rounded-md bg-slate-200" />}
      />
    </section>
  );
}