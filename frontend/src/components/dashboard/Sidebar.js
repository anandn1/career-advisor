export default function Sidebar() {
  return (
    <aside className="glass w-64 min-h-screen p-6 flex flex-col gap-6">
      <h2 className="text-xl font-bold">Dashboard</h2>

      <nav className="flex flex-col gap-4">
        <a href="/dashboard" className="hover:text-primary">Overview</a>
        <a href="/dashboard/analysis" className="hover:text-primary">Skill Gap Analysis</a>
        <a href="/dashboard/interview" className="hover:text-primary">Interview Coach</a>
        <a href="/dashboard/learning-path" className="hover:text-primary">Learning Path</a>
        <a href="/dashboard/profile" className="hover:text-primary">Profile</a>
      </nav>
    </aside>
  );
}
