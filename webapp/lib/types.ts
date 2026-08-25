export interface MatchPrediction {
  home: string;
  away: string;
  date: string;
  prob: { H: number; D: number; A: number };
  pred: "H" | "D" | "A";
  risultato?: string;
  esito?: "H" | "D" | "A";
  ok?: boolean;
  // Quote bookmaker (1X2) — solo se la fonte le porta (football-data)
  bm?: { H: number; D: number; A: number } | null;
  // Probabilità bookmaker normalizzate (overround rimosso)
  bm_prob?: { H: number; D: number; A: number } | null;
  // Suggerimento Kelly sulla giocata con edge migliore (solo future)
  kelly?: {
    pick: "H" | "D" | "A";
    model_p: number;
    odd: number;
    stake_pct: number;
  } | null;
}

export interface KellySummary {
  bankroll_iniziale: number;
  bankroll_finale: number;
  profitto: number;
  roi_pct: number;
  scommesse: number;
  stake_totale_pct: number;
  fraction: number;
}

export interface PredictionsData {
  meta: {
    last_update: string;
    stagione: string;
    ultima_giornata_giocata?: number;
    prossima_giornata?: number;
    nota?: string;
    fonte?: string;
    model_rps?: number;
    bookmaker_rps?: number;
    beats_bookmaker_rps?: boolean;
    kelly?: KellySummary;
  };
  stats: { corrette: number; total: number };
  giornate: Record<string, MatchPrediction[]>;
}
