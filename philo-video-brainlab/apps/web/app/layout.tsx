import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "philo-video-brainlab",
  description:
    "Predict social engagement from TRIBE v2 brain-response trajectories. Recreate the cognitive trajectory that made content work.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="nav">
          <span className="logo">🧠 brainlab</span>
          <Link href="/">Dashboard</Link>
          <Link href="/training-data">Training data</Link>
          <Link href="/intern-guide">Intern guide</Link>
          <Link href="/predict">Pre-post scoring</Link>
          <a href="https://github.com/facebookresearch" target="_blank" rel="noreferrer">
            TRIBE v2
          </a>
        </nav>
        <div className="wrap">{children}</div>
      </body>
    </html>
  );
}
