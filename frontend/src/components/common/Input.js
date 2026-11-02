export default function Input({ label, className="", ...props }) {
  return (
    <div className="w-full flex flex-col gap-1">
      {label && <label className="text-sm text-gray-300">{label}</label>}
      <input
        {...props}
        className={`glass bg-glass-lighter p-3 rounded-xl border-none text-white focus:outline-none ${className}`}
      />
    </div>
  );
}
