// rankTargetProximity.ts
//
// Opens exactly TWO files:
//  • Seed refs .......... <seedId>.json.embedded.json  → vectors for all sources
//  • Selected-source .... <selectedSourceId>.json.embedded.json → vector for TARGET only
//
// The TARGET provides the anchor vector.
// The SELECTED SOURCE is what gets ranked among the seed sources.

import { SummaryResult, WeightsResult } from "../2.Multidimensional"
import { promises as fs } from "fs"
import * as path from "path"

export interface IdTitle {
  arxivId: string
  title: string | null
}

export interface SelectFromRefsResult {
  seed: IdTitle
  sources: IdTitle[]
  selectedSource: IdTitle  // 1-hop
  target: IdTitle | null   // 2-hop anchor
}

export interface RankedItem {
  index?: number
  arxivId: string
  title: string
  score?: number
  reason?: string
  distance?: number
}


export interface RankResult {
  rank: number // 1-based; 0 means selectedSource not in sources
  ordered: Array<RankedItem>
  reflection?: {
    summaryResults: SummaryResult;
    weights: WeightsResult;
}
}

function fileFor(dir: string, arxivId: string): string {
  return path.join(dir, `${arxivId}.json.embedded.json`)
}

async function readJson(file: string): Promise<any | null> {
  try { return JSON.parse(await fs.readFile(file, "utf8")) }
  catch { return null; }
}

function isNumArray(v: any): v is number[] {
  return Array.isArray(v) && v.every((x) => typeof x === "number")
}

// In this schema, references[i].title holds the vector
function vecFromRef(ref: any): number[] | null {
  return isNumArray(ref?.title) ? (ref.title as number[]) : null
}

export const norm = (s: string) => s.trim().toLowerCase().replace(/v\d+$/i, "")

function cosineDist(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`)
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i]
    dot += x * y
    na += x * x
    nb += y * y
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  if (!denom) throw new Error(`Zero-norm vector`)
  return 1 - dot / denom
}

/**
 * Rank all seed sources by proximity to the target,
 * using the target's vector (from selectedSource file) as anchor.
 * The rank reported is for the selectedSource itself among all sources.
 */
export async function rankTargetProximity(
  result: SelectFromRefsResult,
  embeddingsDir = "data/cache/refs/embeddings"
): Promise<RankResult> {
  const { seed, sources, selectedSource, target } = result
  if (!target) throw new Error(`Target missing for seed ${seed.arxivId}.`)

  // 1. Load SEED refs (vectors for all sources)
  const seedFile = fileFor(embeddingsDir, seed.arxivId)
  const seedJson = await readJson(seedFile)
  if (!seedJson) throw new Error(`Seed refs file not found: ${seedFile}`)
  const seedRefs: any[] = Array.isArray(seedJson.references) ? seedJson.references : []
  if (seedRefs.length === 0) throw new Error(`Seed ${seed.arxivId} has no references in ${seedFile}.`)

  const seedVecById = new Map<string, number[]>()
  for (const r of seedRefs) {
    const id = r?.arxivId
    const v = vecFromRef(r)
    if (typeof id === "string" && v) seedVecById.set(norm(id), v)
  }

  // 2. Load SELECTED SOURCE refs (for TARGET anchor)
  const selFile = fileFor(embeddingsDir, selectedSource.arxivId)
  const selJson = await readJson(selFile)
  if (!selJson) throw new Error(`Selected-source refs file not found: ${selFile}`)
  const selRefs: any[] = Array.isArray(selJson.references) ? selJson.references : []
  if (selRefs.length === 0) throw new Error(`Selected source ${selectedSource.arxivId} has no references in ${selFile}.`)

  // Anchor vector = target's vector inside selectedSource.refs
  let targetVec: number[] | null = null
  const tNorm = norm(target.arxivId)
  for (const r of selRefs) {
    const id = typeof r?.arxivId === "string" ? norm(r.arxivId) : "";
    if (id === tNorm) { targetVec = vecFromRef(r); break; }
  }
  if (!targetVec) {
    throw new Error(`Target ${target.arxivId} not found (or not embedded) in selected source ${selectedSource.arxivId} refs.`)
  }

  // 3. Gather usable seed sources
  const loaded: Array<{ arxivId: string; title: string | null; vec: number[] }> = []
  for (const s of sources) {
    const v = seedVecById.get(norm(s.arxivId))
    if (!v) continue
    loaded.push({ arxivId: s.arxivId, title: s.title, vec: v })
  }
  if (loaded.length === 0) {
    throw new Error(`No usable source embeddings for seed ${seed.arxivId}.`)
  }

  // 4. Compute distances to anchor (targetVec)
  const ordered = loaded
    .map(x => ({ arxivId: x.arxivId, title: x.title, distance: cosineDist(x.vec, targetVec!) }))
    .sort((a, b) => a.distance - b.distance)

  // 5. Rank SELECTED SOURCE among the ordered sources
  const idx = ordered.findIndex(x => norm(x.arxivId) === norm(selectedSource.arxivId))
  const rank = idx >= 0 ? idx + 1 : 0

  return { rank, ordered }
}