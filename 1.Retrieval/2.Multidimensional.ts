import { SelectFromRefsResult } from './utils/fetch'
import { GeminiClient } from './utils/gemini'
import { RankResult } from './utils/rank'


const prompt = (refs:SelectFromRefsResult) => `
You will be given a list of research papers with their titles, and ID.

Your will also be given a target research paper.

Your task is to help infer which possible source links to the target paper.

For each paper from the list, you will provide:
a. An explanation of why it might link to the target paper.
b. A contrastive explanation of why it might NOT link to the target paper.

Output your results in JSON format.

Target Paper:
${refs.target.arxivId}: ${refs.target.title}

Source Papers:
${refs.sources.map((s, i) => `${i + 1}. ${s.arxivId}: ${s.title}`).join('\n\n')}
`

const schema = {
    type: 'array',
    items: {
        type: 'object',
        properties: {
            arxivId: { type: 'string' },
            title: { type: 'string' },
            explanation: { type: 'string' },
            contrastive_explanation: { type: 'string' }
        },
        required: ['arxivId', 'title', 'explanation', 'contrastive_explanation']
    }
}

interface ContrastResult {
    arxivId: string
    title: string
    explanation: string
    contrastive_explanation: string
}
export const contrast = async (refs: SelectFromRefsResult): Promise<ContrastResult[]> => {
    return await GeminiClient<ContrastResult[]>(prompt(refs), 'gemini-2.5-flash-lite', schema)
}


const summaryPrompt = (explanations: string[], contrasts: string[], refs: SelectFromRefsResult) => `
You will be given a list of explanations that justify why certain research papers might be linked to a target paper.

You will also be given a list of contrastive explanations that justify why those same research papers might NOT be linked to the target paper.

Your task is to summarize the key common themes from both the explanations and the contrastive explanations.

For each common theme, include a description.

For reference, you will also be provided with the title of the target paper.

Output your results in JSON format.

Target Paper:
${refs.target.arxivId}: ${refs.target.title}



Explanations:
${explanations.map((e, i) => `${i + 1}. ${e}`).join('\n\n')}

Contrastive Explanations:
${contrasts.map((c, i) => `${i + 1}. ${c}`).join('\n\n')}

`

const summarySchema = {
    type: 'object',
    properties: {
        explanation_themes: {
            type: 'array',
            items: {
                type: 'object',
                properties: {
                    theme: { type: 'string' },
                    description: { type: 'string' }
                },
                required: ['theme', 'description']
            }
        },
        contrastive_themes: {
            type: 'array',
            items: {
                type: 'object',
                properties: {
                    theme: { type: 'string' },
                    description: { type: 'string' }
                },
                required: ['theme', 'description']
            }
        }
    },
    required: ['explanation_themes', 'contrastive_themes']
}

export interface SummaryResult {
    explanation_themes: { theme: string, description: string }[]
    contrastive_themes: { theme: string, description: string }[]
}

export const summary = async (refs: SelectFromRefsResult): Promise<SummaryResult> => {
    const contrastResults = await contrast(refs)

    const explanations = contrastResults.map(({ explanation }) => explanation)
    const contrasts = contrastResults.map(({ contrastive_explanation }) => contrastive_explanation)

    const prompt = summaryPrompt(explanations, contrasts, refs)
    return await GeminiClient<SummaryResult>(prompt, 'gemini-2.5-flash-lite', summarySchema)
}


const getPositivePrompt = (summary:SummaryResult, refs: SelectFromRefsResult) => `
Your task is to help find which document references a research paper.
You will be provided with a list of themes that summarize why a given document might be a likely reference.

Your task is to score for each document, based on its title, how well it matches the provided themes.
Output a score from 0 to 10 for each theme, where 0 means the document does not match the specific theme, and 10 means the document matches perfectly the given theme.

Output your results in JSON format.

Themes:
${summary.explanation_themes.map((t, i) => `${i + 1}. ${t.theme}: ${t.description}`).join('\n\n')}

Documents:
${refs.sources.map((s, i) => `${i + 1}. ${s.arxivId}: ${s.title}`).join('\n\n')}
`

const getNegativePrompt = (summary:SummaryResult, refs: SelectFromRefsResult) => `
Your task is to help find which document references a research paper.
You will be provided with a list of themes that summarize why a given document might NOT be a likely reference.

Your task is to score for each document, based on its title, how well it matches the provided themes.
Output a score from 0 to 10 for each theme, where 0 means the document does not match the specific theme, and 10 means the document matches perfectly the given theme.

Output your results in JSON format.

Themes:
${summary.contrastive_themes.map((t, i) => `${i + 1}. ${t.theme}: ${t.description}`).join('\n\n')}

Documents:
${refs.sources.map((s, i) => `${i + 1}. ${s.arxivId}: ${s.title}`).join('\n\n')}
`

const ScoreSchema = {
    type: 'array',
    items: {
        type: 'object',
        properties: {
            arxivId: { type: 'string' },
            title: { type: 'string' },
            scores: {
                type: 'array',
                items: { 
                    type: 'object',
                    properties: {
                        theme: { type: 'string' },
                        score: { type: 'number' }
                    },
                    required: ['theme', 'score']
                }
            }
        },
        required: ['arxivId', 'title', 'scores']
    }
}

export interface ScoreResult {
    arxivId: string
    title: string
    scores: { theme: string, score: number }[]
}

export const evaluate = async (refs: SelectFromRefsResult) => {
    const summaryResults = await summary(refs)

    const positivePrompt = getPositivePrompt(summaryResults, refs)
    const negativePrompt = getNegativePrompt(summaryResults, refs)

    const positiveScores = await GeminiClient<ScoreResult[]>(positivePrompt, 'gemini-2.5-flash-lite', ScoreSchema)
    const negativeScores = await GeminiClient<ScoreResult[]>(negativePrompt, 'gemini-2.5-flash-lite', ScoreSchema)

    return { positiveScores, negativeScores, summaryResults }
}

const weightsPrompt = (summary: SummaryResult, refs: SelectFromRefsResult) => `
You are helping predict which research papers reference a target paper.

You will be provided with a list of themes that have been identified as important for making this prediction.

You will also be provided with the title of the target paper.

Your task is to assign a weight from 1 to 10 for each theme, indicating its importance in predicting whether a document references the target paper.

You will receive two sets of themes: one set that supports the connection (positive themes) and another set that opposes it (negative themes). In both cases, assigning a weigth of 10 means the theme is extremely important, while a weight of 1 means it is barely important at all.

Provide an explanation for each weight you assign.

Output your results in JSON format.

Target Paper:
${refs.target.arxivId}: ${refs.target.title}

Positive Themes:
${summary.explanation_themes.map((t, i) => `${i + 1}. ${t.theme}: ${t.description}`).join('\n\n')}

Negative Themes:
${summary.contrastive_themes.map((t, i) => `${i + 1}. ${t.theme}: ${t.description}`).join('\n\n')}
`

const WeightsSchema = {
    type: 'object',
    properties: {
        positive_weights: {
            type: 'array',
            items: {
                type: 'object',
                properties: {
                    theme: { type: 'string' },
                    weight: { type: 'number' },
                    explanation: { type: 'string' }
                },
                required: ['theme', 'weight', 'explanation']
            }
        },
        negative_weights: {
            type: 'array',
            items: {
                type: 'object',
                properties: {
                    theme: { type: 'string' },
                    weight: { type: 'number' },
                    explanation: { type: 'string' }
                },
                required: ['theme', 'weight', 'explanation']
            }
        }
    },
    required: ['positive_weights', 'negative_weights']
}

export interface WeightsResult {
    positive_weights: { theme: string, weight: number }[]
    negative_weights: { theme: string, weight: number }[]
}

export const score = async (refs: SelectFromRefsResult):Promise<RankResult> => {
    const { positiveScores, negativeScores, summaryResults } = await evaluate(refs)
    const prompt = weightsPrompt(summaryResults, refs)
    const weights = await GeminiClient<WeightsResult>(prompt, 'gemini-2.5-flash-lite', WeightsSchema)
    const scores = getScores({ refs, positiveScores, negativeScores, summaryResults, weights })

    const rank = scores.findIndex(s => s.arxivId === refs.selectedSource.arxivId) + 1
    const reflection = { summaryResults, weights }
    return { rank, ordered: scores, reflection }
}

interface iGetScores {
    weights: WeightsResult
    refs: SelectFromRefsResult
    positiveScores: ScoreResult[]
    negativeScores: ScoreResult[]
    summaryResults: SummaryResult
}

export const getScores = ({ refs, positiveScores, negativeScores, summaryResults, weights }: iGetScores) => {
    const positiveThemes = summaryResults.explanation_themes.map(t => t.theme.toLowerCase())
    const negativeThemes = summaryResults.contrastive_themes.map(t => t.theme.toLowerCase())

    const positiveWeightsMap = positiveThemes.reduce((acc, theme) => {
        const weight = weights.positive_weights.find(w => w.theme.toLowerCase() === theme)?.weight || 5
        return { ...acc, [theme]: weight }
    }, {} as Record<string, number>)

    const negativeWeightsMap = negativeThemes.reduce((acc, theme) => {
        const weight = weights.negative_weights.find(w => w.theme.toLowerCase() === theme)?.weight || 5
        return { ...acc, [theme]: weight }
    }, {} as Record<string, number>)


    const scores = refs.sources.map(source => {
        const pos = positiveScores.find(s => s.arxivId === source.arxivId)
        const positveScore = positiveThemes.reduce((acc, theme) => {
            const scoreObj = pos?.scores.find(s => s.theme.toLowerCase() === theme)
            return acc + ((scoreObj ? scoreObj.score : 0) * (1 + ((positiveWeightsMap[theme] || 5)/10)))
        }, 0)

        const neg = negativeScores.find(s => s.arxivId === source.arxivId)
        const negativeScore = negativeThemes.reduce((acc, theme) => {
            const scoreObj = neg?.scores.find(s => s.theme.toLowerCase() === theme)
            return acc + ((scoreObj ? scoreObj.score : 0) * (1 + ((negativeWeightsMap[theme] || 5)/10)))
        }, 0)

        return {
            arxivId: source.arxivId,
            title: source.title,
            score: positveScore - negativeScore,
            positiveScores: pos,
            negativeScores: neg,
        }
        
    })

    return scores.sort((a, b) => b.score - a.score)
}
