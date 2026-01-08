import { GoogleGenAI } from '@google/genai'
import { writeFileSync } from 'fs'
import { mkdir, writeFile } from 'fs/promises'
import path from 'path'

// Define GeminiSchema interface for validation
interface GeminiSchema {
  type?: string
  properties?: Record<string, GeminiSchema>
  items?: GeminiSchema
  required?: string[]
}

export type GEMMA_MODELS = 'gemma-3-12b-it' | 'gemma-3-27b-it'

interface iGemmaClient {
  prompt: string;
  schema: GeminiSchema;
  model?: GEMMA_MODELS;
}

// --- Validation Logic (Kept as is, it's solid) ---
const validateAgainstSchema = (value: unknown, schema: GeminiSchema): boolean => {
  if (!schema || typeof schema !== 'object') return false

  const typedSchema = schema as {
    type?: string
    properties?: Record<string, GeminiSchema>
    items?: GeminiSchema
    required?: string[]
  }

  switch (typedSchema.type) {
    case 'object': {
      if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
      const required = typedSchema.required ?? []
      for (const key of required) {
        if (!(key in (value as Record<string, unknown>))) return false
      }

      const properties = typedSchema.properties ?? {}
      for (const [key, subschema] of Object.entries(properties)) {
        if (key in (value as Record<string, unknown>)) {
          if (!validateAgainstSchema((value as Record<string, unknown>)[key], subschema)) return false
        }
      }
      return true
    }
    case 'array': {
      if (!Array.isArray(value)) return false
      if (!typedSchema.items) return true
      return value.every(entry => validateAgainstSchema(entry, typedSchema.items as GeminiSchema))
    }
    case 'string':
      return typeof value === 'string'
    case 'number':
    case 'integer':
      return typeof value === 'number'
    case 'boolean':
      return typeof value === 'boolean'
    default:
      return false
  }
}

// --- Extraction Logic ---
function extractJson(text: string): any {
  // 1. Find the first occurrence of both types
  const firstBrace = text.indexOf('{');
  const firstBracket = text.indexOf('[');

  // 2. Determine which one starts the JSON structure
  // If we found a brace, AND (it's before the bracket OR no bracket found) -> It's an Object
  if (firstBrace !== -1 && (firstBracket === -1 || firstBrace < firstBracket)) {
    const lastBrace = text.lastIndexOf('}');
    if (lastBrace > firstBrace) {
      return JSON.parse(text.substring(firstBrace, lastBrace + 1));
    }
  }

  // 3. Otherwise, if we found a bracket -> It's an Array
  if (firstBracket !== -1 && (firstBrace === -1 || firstBracket < firstBrace)) {
    const lastBracket = text.lastIndexOf(']');
    if (lastBracket > firstBracket) {
      return JSON.parse(text.substring(firstBracket, lastBracket + 1));
    }
  }

  throw new Error("No valid JSON object or array found in response");
}

// --- Main Client ---
export const GemmaClient = async <T>({ prompt, schema, model }: iGemmaClient): Promise<T> => {
  const targetModel = model || 'gemma-3-12b-it'
  if (!process.env.GEMINI) throw new Error('GEMINI API key is not configured')

  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI })
  const maxAttempts = 1
  let lastError: unknown = null

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    // FIX: Stronger Prompt Engineering because native JSON mode isn't supported
    const augmentedPrompt = `${prompt}

---
STRICT OUTPUT INSTRUCTIONS:
1. You are a JSON generator. You must output VALID JSON only.
2. Do not include markdown formatting (like \`\`\`json).
3. Do not output any conversational text before or after the JSON.
4. The output must strictly follow this JSON Schema:
${JSON.stringify(schema, null, 2)}
---
`

    try {
      // FIX: Ensure no generationConfig requesting 'application/json' is passed here
      const result = await ai.models.generateContent({
        model: targetModel,
        contents: [{ role: 'user', parts: [{ text: augmentedPrompt }] }],
        // explicit config to avoid defaults implicitly setting mimeTypes
        config: {
          temperature: 0.1, // Low temperature for deterministic structure
        }
      })

      const text = result?.candidates?.[0]?.content?.parts?.[0]?.text || ''
      
      // Attempt parse
      let parsed: any = null
      let valid = false
      
      try {
        parsed = extractJson(text)
        valid = validateAgainstSchema(parsed, schema)
      } catch (e) {
        console.warn(`Attempt ${attempt}: JSON parse failed`, e)
      }

      // Logging logic
      try {
        const dir = 'utils/logs'
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
        const filename = `${timestamp}-gemma-${attempt}.json`
        
        // Ensure dir exists (using simple fallback if specific strict logic needed)
        await mkdir(dir, { recursive: true }).catch(() => {}) 
        
        await writeFile(path.join(dir, filename), JSON.stringify({
          model: targetModel,
          prompt: augmentedPrompt,
          response: text,
          parsed,
          valid,
        }, null, 2))
      } catch (loggingError) {
        console.log('Logging failed, proceeding...', loggingError)
      }

      if (valid && parsed) return parsed as T
      
      if (!valid) {
         lastError = new Error(`Gemma response structure mismatch on attempt ${attempt}`)
      }

    } catch (error) {
      console.log('Error occurred while processing Gemma request:', error)
      lastError = error
    }
  }

  throw lastError instanceof Error ? lastError : new Error('Gemma request failed after max attempts')
}
