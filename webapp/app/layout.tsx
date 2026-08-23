import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Serie A — Predizioni",
  description: "Predizioni CatBoost + Pi-Rating per la Serie A",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="it">
      <body>{children}</body>
    </html>
  );
}
