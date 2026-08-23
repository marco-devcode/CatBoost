export interface MatchPrediction {
  home: string;
  away: string;
  date: string;
  prob: { H: number; D: number; A: number };
  pred: "H" | "D" | "A";
  risultato?: string;
  esito?: "H" | "D" | "A";
  ok?: boolean;
}

export interface PredictionsData {
  meta: {
    last_update: string;
    stagione: string;
    ultima_giornata_giocata?: number;
    prossima_giornata?: number;
    nota?: string;
  };
  stats: { corrette: number; total: number };
  giornate: Record<string, MatchPrediction[]>;
}
