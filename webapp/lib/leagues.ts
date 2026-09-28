// Elenco dei campionati serviti dal sito.
//
// Ogni lega ha il SUO file JSON (`predictions_<key>.json`): i dati dei due
// campionati non devono mai mescolarsi, ne' in UI ne' in storico. Questo
// elenco e' l'unica fonte di verita' sul lato frontend — se aggiungi una
// lega nel backend (`src/leagues.py`) devi aggiungerla qui, altrimenti la
// pagina non la carica.
//
// `webapp/public/*.json` e' servito staticamente: il nome del file non puo'
// contenere caratteri che rompano l'URL.

import { LeagueOption } from "./types";

export const LEAGUES: LeagueOption[] = [
  { key: "seriea", name: "Serie A", json: "/predictions_seriea.json" },
  { key: "eredivisie", name: "Eredivisie", json: "/predictions_eredivisie.json" },
];

export const DEFAULT_LEAGUE = LEAGUES[0];
