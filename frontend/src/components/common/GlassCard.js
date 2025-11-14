export default function GlassCard({ children, className = "" }) {
  return (
    <div className={`glass p-6 rounded-2xl shadow-glass-lg ${className}`}>
      {children}
    </div>
  );
}
