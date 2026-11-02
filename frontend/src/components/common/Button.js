export default function Button({ children, variant = "primary", className = "", ...props }) {
  const base = "px-6 py-3 rounded-xl font-semibold transition-all";

  const variants = {
    primary: "bg-primary text-white hover:bg-primary/80 shadow-lg",
    outline: "border border-white/20 text-white hover:bg-white/10",
  };

  return (
    <button {...props} className={`${base} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
}
