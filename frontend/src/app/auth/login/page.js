import Input from "@/components/common/Input";
import Button from "@/components/common/Button";
import GlassCard from "@/components/common/GlassCard";

export default function LoginPage() {
  return (
    <GlassCard className="max-w-md w-full">
      <h2 className="text-3xl font-bold mb-6">Login</h2>

      <form className="flex flex-col gap-4">
        <Input label="Email" type="email" />
        <Input label="Password" type="password" />

        <Button type="submit">Login</Button>
      </form>

      <p className="mt-4 text-sm text-gray-300">
        Don't have an account? <a href="/auth/register" className="text-primary">Register</a>
      </p>
    </GlassCard>
  );
}
