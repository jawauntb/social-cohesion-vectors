/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ["@brainlab/db"],
  experimental: {
    // allow importing the workspace db package from the monorepo
    externalDir: true,
  },
};

export default nextConfig;
