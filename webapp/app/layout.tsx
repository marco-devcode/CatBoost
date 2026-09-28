import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "CatBoost — Predizioni",
  description:
    "Predizioni CatBoost + Pi-Rating per Serie A ed Eredivisie",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="it">
      <body>
        <Link href="/modello" className="brand-badge">
          CATBOOST
        </Link>
        {children}
      </body>
    </html>
  );
}
