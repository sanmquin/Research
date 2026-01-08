import { readdirSync } from 'fs'

interface iResult {
  references: {
    seed: { arxivId: string; title: string };
    sources: { arxivId: string; title: string }[];
    selectedSource: { arxivId: string; title: string };
    target: { arxivId: string; title: string };
  },

  embeddings: {
    rank: number;
    ordered:       {
      arxivId: string;
      title: string;
      distance: number;
    }[]
  }

  llm: {
    rank: 1,
    ordered: {
      index: number,
      arxivId: string
      title: string
      score: number,
      reason: string
    }[]
  }
}

// ----- Entry point --------------------------------------------------
const main = async () => {
  const dir = 'results/1.Zero-Shot'
  const results = readdirSync(`./${dir}`).filter(f => f.endsWith('.json'))

  const avgRank = { embedding: 0, llm: 0 }
  const wins = { embedding: 0, llm: 0 } 
  const positions = { embedding: [] as number[], llm: [] as number[] }

  for (const file of results) {
    const content = await import(`../${dir}/${file}`) as iResult
    avgRank.embedding += content.embeddings.rank
    avgRank.llm += content.llm.rank
    positions.embedding.push(content.embeddings.rank/content.embeddings.ordered.length)
    positions.llm.push(content.llm.rank/content.llm.ordered.length)

    if (content.embeddings.rank < content.llm.rank) wins.embedding += 1
    else if (content.llm.rank < content.embeddings.rank) wins.llm += 1
  }

  // Who won more often?
  const embeddingWins = wins.embedding
  const llmWins = wins.llm

  console.log(`Embedding wins: ${embeddingWins}, LLM wins: ${llmWins}`)

  const avgEmbeddingRank = avgRank.embedding / results.length
  const avgLlmRank = avgRank.llm / results.length

  console.log(`Average Embedding Rank: ${avgEmbeddingRank.toFixed(2)}`)
  console.log(`Average LLM Rank: ${avgLlmRank.toFixed(2)}`)

  const avgEmbeddingPosition = positions.embedding.reduce((a, b) => a + b, 0) / positions.embedding.length
  const avgLlmPosition = positions.llm.reduce((a, b) => a + b, 0) / positions.llm.length

  console.log(`Average Embedding Position: ${(avgEmbeddingPosition * 100).toFixed(2)}%`)
  console.log(`Average LLM Position: ${(avgLlmPosition * 100).toFixed(2)}%`)
}
  
main().catch((err) => {
  console.error(`Error: ${err.message}`)
  process.exit(1)
});
