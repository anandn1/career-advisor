import { redirect } from "next/navigation";

export default async function SessionIndex({ params }: PageProps<"/s/[sessionId]">) {
  const { sessionId } = await params;
  redirect(`/s/${sessionId}/skills`);
}
