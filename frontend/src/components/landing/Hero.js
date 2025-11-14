import Button from "@/components/common/Button";

export default function Hero() {
  return (
    <section className="max-w-7xl mx-auto px-6 py-28 grid md:grid-cols-2 gap-12 items-center">
      
      {/* LEFT */}
      <div>
        <h1 className="text-6xl font-extrabold leading-tight">
          Your all-in-one <span className="text-primary">AI</span> career platform
        </h1>

        <p className="mt-6 text-lg text-gray-300">
          Skill gap analysis, interview coaching, and personalized learning paths. 
          Everything you need — in one place.
        </p>

        <div className="mt-10 flex gap-4">
          <a href="/dashboard">
            <Button>Go to Dashboard →</Button>
          </a>
          <a href="/auth/login">
            <Button variant="outline">Login</Button>
          </a>
        </div>
      </div>

      {/* RIGHT */}
      <div className="relative">
        <div className="w-full h-96 bg-primary/40 blur-3xl rounded-full absolute -top-10 left-10"></div>
        <div className="w-full h-80 bg-primary/20 blur-2xl rounded-full absolute top-20 right-10"></div>
        <div className="w-full h-72 glass p-10 relative">
          <h3 className="text-2xl font-semibold">AI Career Tools</h3>
          <p className="text-gray-300 mt-3">Skill gap analysis and mock interviews ready to launch.</p>
        </div>
      </div>
    </section>
  );
}
