import type { Metadata, Viewport } from "next";
import SiteHeader from "./components/SiteHeader";
import "./globals.css";

export const metadata: Metadata = {
  title: "CatBoost: Predizioni Serie A ed Eredivisie",
  description:
    "Modello probabilistico CatBoost + Pi-Rating Dixon-Coles con benchmark RPS Bet365 e simulazione Kelly dinamico.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0c0d0e",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="it">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <SiteHeader />
        <div id="main-content" className="app-canvas">
          {children}
        </div>
        <footer className="app-footer">
          <span>Modello CatBoost + Pi-Rating Dixon-Coles (decay 2 anni)</span>
          <span>Dati storici football-data.co.uk · Stime statistiche, non consigli di scommessa.</span>
        </footer>
      </body>
    </html>
  );
}
