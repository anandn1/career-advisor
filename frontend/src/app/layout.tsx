import type { Metadata } from "next";
import { JetBrains_Mono, Overpass, Source_Sans_3 } from "next/font/google";

import { Providers } from "./providers";
import "./globals.css";

// Overpass descends from US highway signage; it carries headings and labels.
const signage = Overpass({ variable: "--font-signage", subsets: ["latin"] });
const body = Source_Sans_3({ variable: "--font-body", subsets: ["latin"] });
const data = JetBrains_Mono({ variable: "--font-data", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Agentic AI Career Advisor",
  description:
    "AI agents read the job market, find your skill gaps, plan courses, write your resume and run mock interviews.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${signage.variable} ${body.variable} ${data.variable} h-full`}>
      <body className="flex min-h-full flex-col">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
