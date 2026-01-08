import { writeFileSync } from 'fs'
import { mkdir, writeFile } from 'fs/promises'

// Define GeminiSchema interface for validation
interface GeminiSchema {
  type?: string
  properties?: Record<string, GeminiSchema>
  items?: GeminiSchema
  required?: string[]
}

// Adjust these to match your local Ollama tags
export type OLLAMA_MODELS = 'phi3' | 'gemma3:4b' | 'gemma3:1b' | 'qwen3:1.7b' | 'qwen3:0.6b'

interface iOllamaClient {
  prompt: string
  schema: GeminiSchema
  model?: OLLAMA_MODELS
}

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

function extractJson(text: string): any {
  const firstBrace = text.indexOf('{')
  const firstBracket = text.indexOf('[')

  if (firstBrace !== -1 && (firstBracket === -1 || firstBrace < firstBracket)) {
    const lastBrace = text.lastIndexOf('}')
    if (lastBrace > firstBrace) {
      return JSON.parse(text.substring(firstBrace, lastBrace + 1))
    }
  }

  if (firstBracket !== -1 && (firstBrace === -1 || firstBracket < firstBrace)) {
    const lastBracket = text.lastIndexOf(']')
    if (lastBracket > firstBracket) {
      return JSON.parse(text.substring(firstBracket, lastBracket + 1))
    }
  }

  throw new Error('No valid JSON object or array found in response')
}

export const OllamaClient = async <T>({ prompt, schema, model }: iOllamaClient): Promise<T> => {
  const targetModel: OLLAMA_MODELS = model || 'gemma3:4b'
  const maxAttempts = 2
  let lastError: unknown = null

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    // With native schema support, the schema goes in `format`.
    // You can still optionally include it in the prompt; here I keep prompt clean.
    const messages = [
      {
        role: 'user' as const,
        content: `${prompt}\n\nReturn only JSON matching the given schema.`
      }
    ]

    try {
      const res = await fetch('http://localhost:11434/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: targetModel,
          messages,
          stream: false,
          thinking: false,
          format: schema,        // <- native JSON schema tools
          options: {
            temperature: 0       // more deterministic for structured output
          }
        })
      })

      if (!res.ok) {
        throw new Error(`Ollama request failed with status ${res.status} ${res.statusText}`)
      }

      const data: any = await res.json()

      // For structured outputs, message.content is JSON string
      let text: string
      const content = data?.message?.content
      if (typeof content === 'string') {
        text = content
      } else if (Array.isArray(content)) {
        // very defensive: join array parts if any
        text = content.map((c) => (typeof c === 'string' ? c : JSON.stringify(c))).join('')
      } else {
        text = content != null ? String(content) : ''
      }

      let parsed: any
      try {
        parsed = JSON.parse(text)
      } catch {
        // fallback if something weird slips through
        parsed = extractJson(text)
      }

      const valid = parsed !== null && validateAgainstSchema(parsed, schema)

      try {
        const dir = 'utils/logs'
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
        const filename = `${timestamp}-ollama.json`
        await mkdir(dir, { recursive: true })
        await writeFile(
          `${dir}/${filename}`,
          JSON.stringify(
            {
              model: targetModel,
              prompt,
              schema,
              messages,
              response: text,
              parsed,
              valid
            },
            null,
            2
          )
        )
      } catch (loggingError) {
        console.log('Error occurred while writing Ollama log file:', loggingError)
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
        const filename = `ollama-fallback-log-${timestamp}.json`

        writeFileSync(
          filename,
          JSON.stringify(
            {
              model: targetModel,
              prompt,
              schema,
              messages,
              response: text,
              parsed,
              valid
            },
            null,
            2
          )
        )
      }

      if (valid) return parsed as T
      lastError = new Error('Ollama response failed schema validation')
    } catch (error) {
      console.log('Error occurred while processing Ollama request:', error)
      lastError = error
    }
  }

  throw lastError instanceof Error ? lastError : new Error('Ollama request failed')
}
