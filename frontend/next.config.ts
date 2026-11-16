import type { NextConfig } from "next";

// In Docker the browser only talks to this server, which forwards /api/* to the backend
// (one address to open or share, no CORS). Read at build time; unset in local development,
// where the browser calls NEXT_PUBLIC_API_URL directly.
const apiProxyTarget = process.env.API_PROXY_TARGET;

// The proxy drops a request after this long without data. Run streams send a ping every
// 15 s, but an interview answer or a resume upload can wait longer on the LLM.
const PROXY_IDLE_TIMEOUT_MS = 5 * 60 * 1000;

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  output: "standalone",
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  experimental: {
    proxyTimeout: PROXY_IDLE_TIMEOUT_MS,
  },
  async rewrites() {
    if (!apiProxyTarget) return [];
    return [{ source: "/api/:path*", destination: `${apiProxyTarget}/api/:path*` }];
  },
};

export default nextConfig;
