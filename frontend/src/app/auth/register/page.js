import Input from "@/components/common/Input";
import Button from "@/components/common/Button";
import GlassCard from "@/components/common/GlassCard";

export default function RegisterPage() {
  return (
    <GlassCard className="max-w-md w-full">
      <h2 className="text-3xl font-bold mb-6">Create Account</h2>

      <form className="flex flex-col gap-4">
        <Input label="Full Name" />
        <Input label="Email" type="email" />
        <Input label="Password" type="password" />

        <Button type="submit">Register</Button>
      </form>

      <p className="mt-4 text-sm text-gray-300">
        Already have an account? <a href="/auth/login" className="text-primary">Login</a>
      </p>
    </GlassCard>
  );
}
