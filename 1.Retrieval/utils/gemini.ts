import { GoogleGenAI } from '@google/genai'

/**
 * Shared Gemini helper used by the scoring pipeline whenever a structured JSON
 * response is required. Keeping the client logic centralised avoids duplicating
 * retry handling across multiple steps.
 */

export type GEMINI_MODELS = 'gemini-2.5-pro' | 'gemini-2.5-flash' | 'gemini-2.5-flash-lite'
const sleep = (seconds: number) => new Promise(resolve => setTimeout(resolve, seconds * 1000))

type GeminiSchema = Record<string, unknown>

export const GeminiClient = async <T>(prompt: string, model?: GEMINI_MODELS, schema?: GeminiSchema): Promise<T> => {
  model = model || 'gemini-2.5-flash-lite'
  if (!process.env.GEMINI) throw new Error('GEMINI API key is not configured')

  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI })
  let response = null

  try {
    const config = schema ? { responseMimeType: 'application/json', responseSchema: schema } : {}

    // Build parts array - include file if provided
    const parts: any[] = [{ text: prompt }]
    const result: any = await ai.models.generateContent({
      model,
      contents: [{ role: 'user', parts }],
      config
    })

    const text = result?.candidates?.[0]?.content?.parts?.[0]?.text
    response = schema ? JSON.parse(text) : text

    if(model === 'gemini-2.5-flash-lite') await sleep(5)
    else await sleep(3)

    return response as T
  } catch (error) {
    console.log("Error occurred while processing request:", error)
    console.log('Result:', response)
    throw error
  }
}