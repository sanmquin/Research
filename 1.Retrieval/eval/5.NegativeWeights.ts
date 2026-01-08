import { readdirSync } from 'fs'

interface iResult {
  selectedSource: { arxivId: string; title: string };
  target: { arxivId: string; title: string };

  scores: { // LLM
    rank: number
    ordered: {
      arxivId: string,
      title: string,
      positiveScores: {
        arxivId: string
        title: string
        scores: {
          theme: string
          score: number
        }[]
      }
      negativeScores: {
        arxivId: string
        title: string
        scores: {
          theme: string
          score: number
        }[]
      }
      score: number
    }[]
  }

  ranking: { // Embedding
    rank: number;
    ordered:       {
      arxivId: string;
      title: string;
      distance: number;
    }[]
  }
}

// ----- Entry point --------------------------------------------------
const main = async () => {
  const dir = 'results/3.Weights'
  const results = readdirSync(`./${dir}`).filter(f => f.endsWith('.json'))

  const avgRank = { embedding: 0, llm: 0 }
  const wins = { embedding: 0, llm: 0, ties: 0 } 
  const positions = { embedding: [] as number[], llm: [] as number[] }

	// Multi-factor evaluation
  for (const file of results) {
		try {
			const content = await import(`../${dir}/${file}`) as iResult

			// Order scores by positive score
			const llmRank = content.scores.rank
			console.log(`${content.selectedSource.arxivId}: LLM Rank = ${llmRank}, Embedding Rank = ${content.ranking.rank}`)

			avgRank.embedding += content.ranking.rank
			avgRank.llm += llmRank
			positions.embedding.push(content.ranking.rank/content.ranking.ordered.length)
			positions.llm.push(llmRank/content.scores.ordered.length)

			if (content.ranking.rank < llmRank) wins.embedding += 1
			else if (llmRank < content.ranking.rank) wins.llm += 1
			else wins.ties += 1
			
		} catch (err) {
			console.error(`Error processing file ${file}: ${(err as Error).message}`)
		}
  }

  // Who won more often?
  const embeddingWins = wins.embedding
  const llmWins = wins.llm

  console.log(`Embedding wins: ${embeddingWins}, LLM wins: ${llmWins}, Ties: ${wins.ties}`)

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
