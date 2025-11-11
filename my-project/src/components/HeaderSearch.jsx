import logo1 from "/Users/emmawarren/Desktop/VIP-UI/my-project/src/assets/logo2.png"

export default function HeaderSearch() {
  return (
    <header className="bg-gradient-to-b from-[#5aa56a] to-[#4f8f5c]">
      <div className="mx-auto max-w-6xl px-4 py-5">
        <div className="flex items-center gap-6 text-xs text-white/90">
          <img className="inline-block h-15 w-15 rounded-full ring-2 ring-white"
            src={logo1}
          />
          <div className="ml-auto flex items-center gap-6">
            <a href="#" className="hover:underline justify-end">Logout</a>
            <a href="#" className="hover:underline justify-end">Upload Resume</a>
          </div>
        </div>
        <div className="grid grid-cols-12 gap-2">
  
        </div>
      </div>
    </header>
  );
}