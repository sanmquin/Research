import { OLLAMA_MODELS, OllamaClient } from "./utils/ollama"
import { iFeature, iVideo } from "./utils/types"
import { GemmaClient } from "./utils/gemma"
import { writeFile } from "fs/promises"


// 3. Rank training videos using features.
const rankingPrompt = (video: iVideo[], features: iFeature[]) => `
You are a data analyst tasked helping to predict the performance of YouTube videos.

You will be provided with a list of YouTube videos for a given channel.

You will also be provided with a set of features identified as important for video performance.

For each video, please provide a score from 0 to 10 for each feature.

Each feature contains a name, summary, and description.

Format your response as JSON with the video title, and an array of feature scores.

Videos:
${video.map(v => `- ${v.title}`).join('\n')}


Features:
${features.map((f, i) => `${i + 1}. ${f.name} (${f.summary}): ${f.description}`).join('\n')}

`

const rankingSchema = (features:iFeature[]) => {
    const featureNames = features.map(f => f.name)

    return {
        type: 'array',
        items: {
            type: 'object',
            properties: {
                title: { type: 'string' },
                features: {
                    type: 'object',
                    properties: featureNames.reduce((acc, name) => {
                        acc[name] = { type: 'number', minimum: 0, maximum: 10 }
                        return acc
                    }, {}),
                    required: featureNames,
                },
            },
            required: ['title', 'features'],
        }
    }
}


interface iRanking {
    title: string;
    features: Record<string, number>;
}

export const rankVideos = async (videos: iVideo[], features: iFeature[], ollamaModel?: OLLAMA_MODELS) => {
    const batchSize = ollamaModel ? 10 : features.length === 1 ? 40 : 20
    const rankings: iRanking[] = []
    const MAX_TRIES = 3
    let failures = 0

    while(rankings.length < videos.length){
        const unrankedVideos = videos.filter(v => !rankings.find(r => r.title === v.title))
//        console.log(`Ranking ${unrankedVideos.length} videos remaining...`);
        if(unrankedVideos.length === 0) break;
        else if(unrankedVideos.length < batchSize) {
            if(failures >= MAX_TRIES) throw new Error('Max ranking retries exceeded.')
            failures++
        }

        const batch = unrankedVideos.slice(0, batchSize)
        const prompt = rankingPrompt(batch, features)
        const schema = rankingSchema(features)

        if(ollamaModel) {
            const ollamaInput = { prompt, model: ollamaModel, schema }
            const batchRankings = await OllamaClient<iRanking[]>(ollamaInput)
            
            const validRankings = batchRankings.filter(br => batch.find(v => 
                v.title.normalize('NFKC') === br.title.normalize('NFKC')
            ))
            rankings.push(...validRankings)
            continue
        }

        const model = 'gemma-3-12b-it' as const
        const geminiInput = { prompt, schema, model }
        const batchRankings = await GemmaClient<iRanking[]>(geminiInput)

        const validRankings = batchRankings.filter(br => batch.find(v => 
            v.title.normalize('NFKC') === br.title.normalize('NFKC')
        ))
        rankings.push(...validRankings)
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    await writeFile(`./llm/data/rankings-${timestamp}.json`, JSON.stringify(rankings, null, 2))
    return rankings
}

