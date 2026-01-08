// actors/actorRank.ts
// First Reflection Actor – integrates GeminiClient and Rank.ts output.

import { norm, type RankResult, type SelectFromRefsResult } from "../utils/rank";
import { GeminiClient } from "../utils/gemini";

export interface ActorInput {
  start: { arxivId: string; title: string };
  target: { arxivId: string; title: string; abstract?: string };
  rankResult: RankResult;  // output of rankTargetProximity()
}

export interface RankedItem {
  index: number;
  arxivId: string;
  title: string;
  score: number;
  reason: string;
}

export interface ActorMetrics {
  completeness: number;
  semanticCorrelation: number | null;
  correctness: number | null;
}

export interface ActorOutput {
  rank: number;
  ranked: RankedItem[];
  metrics: ActorMetrics;
  raw: any;
}

const ACTOR_SCHEMA = {
  type: "object",
  properties: {
    ranked: {
      type: "array",
      items: {
        type: "object",
        properties: {
          index: { type: "integer" },
          arxivId: { type: "string" },
          title: { type: "string" },
          score: { type: "number" },
          reason: { type: "string" },
        },
        required: ["index", "arxivId", "title", "score", "reason"],
      },
    },
  },
  required: ["ranked"],
}

export const verifier = async (input: SelectFromRefsResult, rankResult: RankResult): Promise<ActorOutput> => {
  const { seed:start, target, selectedSource } = input;
  const options = rankResult.ordered
    .map((r, i) => `${i + 1}. ${r.title} [${r.arxivId}] (distance=${r.distance.toFixed(3)})`)
    .join("\n");

  const prompt = `
You are the ACTOR in the Reflection experiment.
Task: Rank all 1-hop papers from the START to reach the TARGET.

START: ${start.title} [${start.arxivId}]
TARGET: ${target.title} [${target.arxivId}]

Below are ${rankResult.ordered.length} candidate papers (each is a 1-hop citation):
${options}

Rank ALL papers from most to least likely bridge to the target.
Score each from 1–10, justify briefly.
Return ONLY valid JSON per schema.
`;

  const raw = await GeminiClient<{ ranked: RankedItem[] }>(prompt, 'gemini-2.5-flash-lite', ACTOR_SCHEMA);
  const ranked = raw.ranked || [];

  // Compute metrics
  const allIds = rankResult.ordered.map(x => x.arxivId);
  const returnedIds = ranked.map(r => r.arxivId);
  const completeness = returnedIds.filter(id => allIds.includes(id)).length / allIds.length;

  const paired = ranked
    .filter(r => allIds.includes(r.arxivId))
    .map(r => ({
      score: r.score,
      invDist: 1 / (rankResult.ordered.find(o => o.arxivId === r.arxivId)?.distance || 1e-9),
    }));

  const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;
  const [mx, my] = [mean(paired.map(p => p.score)), mean(paired.map(p => p.invDist))];
  const num = paired.reduce((s, p) => s + (p.score - mx) * (p.invDist - my), 0);
  const den = Math.sqrt(
    paired.reduce((s, p) => s + (p.score - mx) ** 2, 0) *
    paired.reduce((s, p) => s + (p.invDist - my) ** 2, 0)
  );
  const semanticCorrelation = den ? num / den : null;

  const topActor = ranked[0]?.arxivId;
  const topEmbed = rankResult.ordered[0]?.arxivId;
  const correctness = topActor === topEmbed ? 1 : 0;

  const idx = ranked.findIndex(x => norm(x.arxivId) === norm(selectedSource.arxivId));
  const rank = idx >= 0 ? idx + 1 : 0;  

  return {
    rank,
    ranked,
    metrics: { completeness, semanticCorrelation, correctness },
    raw,
  };
};
