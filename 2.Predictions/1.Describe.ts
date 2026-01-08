import { OLLAMA_MODELS, OllamaClient } from './utils/ollama'
import { iPrediction, iVideo } from './utils/types'
import { formatViews } from './utils/utils'
import { GemmaClient } from "./utils/gemma"
import { writeFile } from 'fs/promises'


// 1. Select top 20, and bottom 20 videos.
const getVideos = (videos:iVideo[]) => {
    const sortedVideos:iVideo[] = videos.slice().sort((a, b) => b.views - a.views)
    const topVideos = sortedVideos.slice(0, 20)
    const bottomVideos = sortedVideos.slice(-20)
    return { topVideos, bottomVideos }
} // TODO: Select based on expected performance.


const contrastPrompt = (topVideos: iVideo[], bottomVideos: iVideo[]) => `
You are a data analyst tasked with understanding why YouTube titles perform better than others. 

You will be provided with two lists of YouTube titles for a given channel.

One list contains the 20 top performing titles (based on view counts)

The second list contains the 20 bottom performing titles.

Your task is to identify the key features in the title that contribute the most to performance.

For each feature, provide a name for the feature, a one-sentence summary, and a detailed description.

The description should include examples that illustrate why the feature is important.

Please provide exactly 5 distinct features that are mutually exclusive and collectively exhaustive.

Format your response as a JSON array

Top videos:
${topVideos.map(v => `- ${v.title} (${formatViews(v.views)} views)`).join('\n')}

Bottom videos:
${bottomVideos.map(v => `- ${v.title} (${formatViews(v.views)} views)`).join('\n')}
`

const featureSchema = {
    type: 'object',
    properties: {
        name: { type: 'string' },
        summary: { type: 'string' },
        description: { type: 'string' },
    },
    required: ['name', 'summary', 'description'],
}

const contrastSchema = {
    type: 'array',
    items: featureSchema
}


interface iFeature {
    name: string;
    summary: string;
    description: string;
}


// 2. Ask LLM to reason about why the top 20 performed better.
export const getFeatures = async (videos:iVideo[], ollamaModel?:OLLAMA_MODELS) => {
    const { topVideos, bottomVideos } = getVideos(videos)
    const prompt = contrastPrompt(topVideos, bottomVideos)
    const model = 'gemma-3-27b-it' as const

    if(ollamaModel) {
        const ollamaInput = { prompt, model: ollamaModel, schema: contrastSchema }
        const features = await OllamaClient<iFeature[]>(ollamaInput)
        return features
    }

    const geminiInput = { prompt, schema: contrastSchema, model }
    const features = await GemmaClient<iFeature[]>(geminiInput)

    try {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
        await writeFile(`./llm/data/features-${timestamp}.json`, JSON.stringify(features, null, 2)) 
    } catch (error) {}
    
    
    return features
}



const updatePrompt = (features:iFeature[], predictions:iPrediction[], failedFeatures:iFeature[]) => `
You are a senior data analyst reviewing a model that predicts YouTube video title performance.

Your task is to improve the model by identifying, and adding the most critical missing feature.

You will be provided with the existing set of features based on YouTube titles used by the model.

The goal is to identify an additional title feature that is not currently included, but which would significantly enhance the model's predictive accuracy.

To help you with this task, you will be provided with a list of the model's worst predictions.

The predictions will be split into two lists: those that underestimated performance, and those that overestimated performance.

You will also be provided with a list of failed title features that were previously suggested but did not improve the model.

Provide your response in JSON format with the name, summary, and description of the missing feature.


Existing features:
${features.map((f, i) => `${i + 1}. ${f.name} (${f.summary}): ${f.description}`).join('\n')}

Worst predictions:
${predictions.filter(({ delta }) => delta < 0).map(video => 
`- "${video.title}": Actual views = ${formatViews(video.actual)}, Predicted views = ${formatViews(video.predicted)}`).join('\n')}

${predictions.filter(({ delta }) => delta > 0).map(video => 
`- "${video.title}": Actual views = ${formatViews(video.actual)}, Predicted views = ${formatViews(video.predicted)}`).join('\n')}

Failed features:
${failedFeatures.map((f, i) => `${i + 1}. ${f.name} (${f.summary}): ${f.description}`).join('\n')}
`


export const addFeature = async (features:iFeature[], predictions:iPrediction[], failedFeatures:iFeature[]) => {
    const prompt = updatePrompt(features, predictions, failedFeatures)
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    writeFile(`./llm/data/feature-prompt-${timestamp}.json`, JSON.stringify({ features, failedFeatures, predictions }, null, 2))
    const geminiInput = { prompt, schema: featureSchema, model: 'gemma-3-27b-it' as const }
    const newFeature = await GemmaClient<iFeature>(geminiInput)

    await writeFile(`./llm/data/feature-${timestamp}.json`, JSON.stringify(newFeature, null, 2))
    return newFeature
}
