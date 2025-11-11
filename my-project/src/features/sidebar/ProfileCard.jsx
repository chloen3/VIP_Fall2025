import headshot from '/Users/emmawarren/Desktop/VIP-UI/my-project/src/assets/headshot.jpeg'
import { useNavigate } from "react-router-dom"

export default function ProfileCard() {
  const navigate = useNavigate();
  return (
    <section className="ui-card">
      <h3 className="text-lg font-semibold text-[#416a4a]">Welcome back, Avery!</h3>
      <div className="mt-4 flex items-center justify-center">
        <img
          src={headshot}
          alt="Avery's Profile Picture"
          className="h-28 w-28 rounded-full object-cover object-center bg-slate-200"
        />
      </div>


      <button 
        onClick={() => navigate("/profile")}
        className="mt-4 w-full rounded-full bg-[#5aa56a] px-4 py-2 text-sm font-medium text-white hover:bg-[#4b8c58]">
        Edit Profile
      </button>
    </section>
  );
}