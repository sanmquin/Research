// actors/actorRank.ts
// First Reflection Actor – integrates GeminiClient and Rank.ts output.

import { norm, RankedItem, RankResult } from "./utils/rank";
import { SelectFromRefsResult } from "./utils/fetch";
import { GeminiClient } from "./utils/gemini";


const ACTOR_SCHEMA = {
  type: "object",
  properties: {
    ranked: {
      type: "array",
      items: {
        type: "object",
        properties: {
          arxivId: { type: "string" },
          title: { type: "string" },
          score: { type: "number" },
          reason: { type: "string" },
        },
        required: ["arxivId", "title", "score", "reason"],
      },
    },
  },
  required: ["ranked"],
}

export const rankWithActor = async (input: SelectFromRefsResult): Promise<RankResult> => {
  const { seed, sources, target, selectedSource } = input;
  const options = sources.map((r, i) => `${i + 1}. [${r.arxivId}]: ${r.title} `).join("\n");

  const prompt = `
You are the ACTOR in the Reflection experiment.
Task: Rank all 1-hop papers from the START to reach the TARGET.

START: ${seed.title} [${seed.arxivId}]
TARGET: ${target.title} [${target.arxivId}]

Below are ${sources.length} candidate papers (each is a 1-hop citation):
${options}

Rank ALL papers from most to least likely bridge to the target.
Score each from 1–10, justify briefly.
Return ONLY valid JSON per schema.
`

/* """
Predict which paper cites the target.
Order all sources from the most likely to the least likely.

SEED: [${seed.arxivId}] ${seed.title} 
TARGET: [${target.arxivId}] ${target.title}

Below are all ${sources.length} sources from the seed paper:
${options}

Order ALL papers from most to least likely bridge to the target.
Score each from 1–10, justify briefly.
Return ONLY valid JSON per schema.
`;
""" */

  const { ranked } = await GeminiClient<{ ranked: RankedItem[] }>(prompt, 'gemini-2.5-flash-lite', ACTOR_SCHEMA);
  const idx = ranked.findIndex(x => norm(x.arxivId) === norm(selectedSource.arxivId));
  const rank = idx >= 0 ? idx + 1 : 0;

  return { rank, ordered: ranked };
}