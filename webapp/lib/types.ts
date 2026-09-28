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
  // Feature usate dal modello (per trasparenza), in ordine FEATURE_ORDER.
  // Il valore puo' essere null: il valore mancante (squadra con meno di 5
  // partite con statistiche) non e' rappresentabile in JSON come NaN, quindi
  // la pipeline lo scrive come null.
  features?: Record<string, number | null>;
}

export interface KellyBet {
  n: number;
  giornata?: string;
  home: string;
  away: string;
  date: string;
  pick: "H" | "D" | "A";
  pick_label: "Casa" | "Pareggio" | "Trasferta";
  odd: number;
  model_p: number;
  bm_p: number;
  kelly_pct: number;
  stake: number;
  esito?: "H" | "D" | "A";
  win: boolean;
  payout: number;
  delta: number;
  bankroll_dopo: number;
}

export interface KellySummary {
  bankroll_iniziale: number;
  bankroll_finale: number;
  profitto: number;
  roi_pct: number;
  scommesse: number;
  vinte?: number;
  perse?: number;
  stake_totale_pct: number;
  fraction: number;
  // Dettaglio di ogni puntata effettivamente simulata
  scommesse_dettaglio?: KellyBet[];
}

export interface PredictionsData {
  meta: {
    last_update: string;
    stagione: string;
    // Campionato: presente da quando il repo serve piu' leghe. Il JSON e'
    // per lega (predictions_<lega>.json) e questo campo identifica quale.
    league?: string;
    league_name?: string;
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

/** Campionato selezionabile in UI. I JSON sono uno per lega. */
export interface LeagueOption {
  key: string;
  name: string;
  json: string;
}
