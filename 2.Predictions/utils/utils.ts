import { iVideo } from "../types";

import { mkdir, writeFile } from 'fs/promises'
import { GoogleGenAI, File } from '@google/genai'
import MLR from 'ml-regression-multivariate-linear'
import { writeFileSync } from "fs";

/**
 * Shared Gemini helper used by the scoring pipeline whenever a structured JSON
 * response is required. Keeping the client logic centralised avoids duplicating
 * retry handling across multiple steps.
 */

export type GEMINI_MODELS = 'gemini-2.5-pro' | 'gemini-2.5-flash' | 'gemini-2.5-flash-lite'
const sleep = (seconds: number) => new Promise(resolve => setTimeout(resolve, seconds * 1000))

type GeminiSchema = Record<string, unknown>

interface iGeminiClient {
  prompt: string;
  model?: GEMINI_MODELS;
  schema?: GeminiSchema;
  attachments?: File[];
}
export const GeminiClient = async <T>({
  prompt,
  model,
  schema,
  attachments
}: iGeminiClient): Promise<T> => {
  const effectiveModel = model || 'gemini-2.5-flash-lite'
  if (!process.env.GEMINI) throw new Error('GEMINI API key is not configured')

  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI })
  let result: any = null

  const config = schema
    ? { responseMimeType: 'application/json', responseSchema: schema }
    : {}

  const parts: any[] = [{ text: prompt }]
  const seenUris = new Set<string>()

  for (const file of attachments || []) {
    if (!file) continue

    const candidate = file as File & { name?: string; mimeType?: string }
    const uri =
      typeof candidate.uri === 'string' && candidate.uri.trim()
        ? candidate.uri.trim()
        : ''
    const fallbackName =
      typeof candidate.name === 'string' && candidate.name.trim()
        ? candidate.name.trim()
        : ''
    const fileUri = uri || fallbackName
    if (!fileUri || seenUris.has(fileUri)) continue
    seenUris.add(fileUri)

    const mimeType =
      (candidate.mimeType && candidate.mimeType.trim()) ||
      (typeof (file as any).mimeType === 'string' &&
        (file as any).mimeType.trim()) ||
      'application/octet-stream'

    parts.push({ fileData: { fileUri, mimeType } })
  }

  try {
    // First: call the model, but normalize the weird Error-constructor case
    try {
      result = await ai.models.generateContent({
        model: effectiveModel,
        contents: [{ role: 'user', parts }],
        config
      })
    } catch (err: any) {
      if (typeof err === 'function' && err.name === 'Error') {
        const wrapped = new Error(
          '[GeminiClient] underlying library threw Error constructor instead of Error instance'
        )
        // keep the original thing for debugging
        ;(wrapped as any).original = err
        console.error('Weird Gemini error:', err)
        throw wrapped
      }
      throw err
    }

    const text = result?.candidates?.[0]?.content?.parts?.[0]?.text
    let response: any = null

    try {
      response = schema ? JSON.parse(text) : text
    } catch (err) {
      console.log(text)
      throw new Error(
        `[GeminiClient] Failed to parse JSON response from Gemini model ${effectiveModel}`
      )
    }

    if (effectiveModel === 'gemini-2.5-flash-lite') await sleep(5)
    else await sleep(3)

    const dir = 'utils/logs'
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    const filename = `${timestamp}.json`
    await mkdir(dir, { recursive: true })
    await writeFile(
      `${dir}/${filename}`,
      JSON.stringify(
        {
          model: effectiveModel,
          prompt,
          response
        },
        null,
        2
      )
    )

    return response as T
  } catch (error) {
    console.log('Error occurred inside GeminiClient:', error)
    console.log('Result (if any):', result)
    // DO NOT swallow it — propagate up so your top-level catch sees a real Error
    if (error instanceof Error) {
      throw error
    }
    throw new Error(
      '[GeminiClient] Non-Error value thrown',
    )
  }
}

export const uploadFile = async (
  content: Buffer,
  mimeType: string,
  displayName?: string
): Promise<File> => {
  if (!process.env.GEMINI) throw new Error('GEMINI API key is not configured');

  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI });

  // Buffer → ArrayBuffer slice → Blob (this matches the docs)
  const arrayBuffer = content.buffer.slice(
    content.byteOffset,
    content.byteOffset + content.byteLength
  ) as ArrayBuffer;

  const blob = new Blob([arrayBuffer], { type: mimeType });

  console.log(`Uploading PDF buffer to Gemini: ${displayName}`);

  const uploadedFile = await ai.files.upload({
    file: blob,
    config: {
      displayName: displayName ?? 'pdf-upload',
    },
  });

  // IMPORTANT: wait until Gemini finishes processing the PDF
  let processed = await ai.files.get({ name: uploadedFile.name });
  while (processed.state === 'PROCESSING') {
    console.log(`current file status: ${processed.state}, retrying...`);
    await new Promise((resolve) => setTimeout(resolve, 5000));
    processed = await ai.files.get({ name: uploadedFile.name });
  }

  if (processed.state === 'FAILED') {
    throw new Error(
      `Gemini file processing failed: ${processed.error?.message ?? 'unknown error'}`
    );
  }

  console.log(`File ready: ${processed.name} (${processed.state})`);

  // Normalize what your code expects back
  return {
    name: processed.name || '',
    displayName: processed.displayName,
    mimeType: processed.mimeType,
    uri: processed.uri || processed.name,
  };
};


/**
 * Delete a file from Gemini Files API
 * @param fileName - Name of the file to delete (e.g., 'files/abc123')
 */
export const deleteFile = async (fileName: string): Promise<void> => {
  if (!process.env.GEMINI) throw new Error('GEMINI API key is not configured')
  
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI })
  
  console.log(`Deleting file from Gemini: ${fileName}`)
  await ai.files.delete({ name: fileName })
  console.log(`File deleted successfully: ${fileName}`)
}


interface iEmbeddingResponse { embeddings: number[][] }
interface iEmbeddedText { text: string; embeddings: number[] }
export const embedTexts = async(texts: string[]): Promise<iEmbeddedText[]> => {
    const timestart = Date.now()
    const headers = { 'Content-Type': 'application/json' }
    const request = { method: 'POST', headers, body: JSON.stringify({ texts }) }
    const response = await fetch('http://localhost:8000/embed/texts', request)
    const { embeddings }:iEmbeddingResponse = await response.json()

    const errorMsg = 'Embedding service returned an error'
    if (!embeddings || embeddings.length !== texts.length) throw new Error(errorMsg)
    const timeend = Date.now()
    console.log(`Embedded ${texts.length} texts in ${(timeend - timestart) / 1000} seconds.`)

    return embeddings.map((embeddings:number[], i:number) => ({ text: texts[i], embeddings }))
}

const RECENT_VIDEOS_COUNT = 5
export const getRecentVideos = (videos:iVideo[]) => videos.map((v:iVideo, i) => {
        // Use views from the last 5 videos as features.
        const start = Math.max(0, i - RECENT_VIDEOS_COUNT)
        const recentViews = videos.slice(start, i).map(v2 => Math.log(v2.views + 1))
        return { video: v, recentViews}
    }).filter(({ recentViews }) => recentViews.length === RECENT_VIDEOS_COUNT) // Remove first 5 videos with no features.

export const formatViews = (v: number): string => {
  if (v < 1_000) return v.toFixed(0)
  if (v < 1_000_000) return (v / 1_000).toFixed(1) + "K"
  return (v / 1_000_000).toFixed(1) + "M"
}

export interface iResults {
    totalLogError: number;
    totalViewsError: number;
    averageLogError: number;
    averageViewsError: number;
    typicalFactor: number;
    averagePredictedViews: number;
    averageActualViews: number;
    videoCount: number;
    mlr: string[];
}

/**
 * predLog: predicted log(values)
 * actualLog: actual log(values)
 */
export const reportLogErrors = (predLog: number[], actualLog: number[], mlr?:MLR, log=false): iResults => {
  if (predLog.length !== actualLog.length) {
    throw new Error(`Length mismatch: pred=${predLog.length}, actual=${actualLog.length}`)
  }

  const n = predLog.length

  let sumAbsLogError = 0
  let sumAbsViewsError = 0
  let sumPredViews = 0
  let sumActualViews = 0

  for (let i = 0; i < n; i++) {
    const p = predLog[i]
    const a = actualLog[i]

    const predViews = Math.exp(p)
    const actualViews = Math.exp(a)

    sumPredViews += predViews
    sumActualViews += actualViews

    const logError = Math.abs(p - a)
    const viewsError = Math.abs(predViews - actualViews)

    sumAbsLogError += logError
    sumAbsViewsError += viewsError
  }

  const avgLogError = sumAbsLogError / n
  const avgViewsError = sumAbsViewsError / n
  const typicalFactor = Math.exp(avgLogError)

  const avgPredViews = sumPredViews / n
  const avgActualViews = sumActualViews / n

  if(log) console.log(`Average error: ${formatViews(avgViewsError)} (~x${typicalFactor.toFixed(2)}) over ${n} videos.`)

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
  const results = {
      totalLogError: sumAbsLogError,
      totalViewsError: sumAbsViewsError,
      averageLogError: avgLogError,
      averageViewsError: avgViewsError,
      typicalFactor,
      averagePredictedViews: avgPredViews,
      averageActualViews: avgActualViews,
      videoCount: n,
      mlr: mlr ? mlr.weights.map(w => w[0].toFixed(4)) : undefined
  }

  const fileName = `./utils/results/evaluation-${timestamp}.json`
  writeFileSync(fileName, JSON.stringify(results, null, 2))

  return results
}
