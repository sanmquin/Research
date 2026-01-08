// selectFromRefs.ts
//
// Seed (paper) → Sources (seed.references)
// Pick a RANDOM source that itself has refs → selectedSource (1-hop)
// From selectedSource.references pick a RANDOM paper → target (2-hop)

import { promises as fs } from "fs"
import * as path from "path"

export interface IdTitle {
  arxivId: string
  title: string | null
}

interface PaperSchema { title?: string; abstract?: string; }
interface ReferenceItem { arxivId: string; title?: string; }
interface ReferenceSchema {
  sourceArxivId: string
  references: ReferenceItem[]
}

export interface SelectFromRefsOptions {
  refsDir?: string   // default "data/cache/refs"
  papersDir?: string // default "data/cache/papers"
}

export interface SelectFromRefsResult {
  seed: IdTitle
  sources: IdTitle[]
  selectedSource: IdTitle // random first-hop (must have refs)
  target: IdTitle | null         // random second-hop from selectedSource.refs
}

async function readJsonIfExists<T>(filepath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filepath, "utf8")
    return JSON.parse(raw) as T
  } catch (err: any) {
    if (err?.code === "ENOENT" || err?.code === "ENOTDIR") return null
    throw err
  }
}

async function loadRefs(id: string, refsDir: string): Promise<ReferenceSchema | null> {
  return readJsonIfExists<ReferenceSchema>(path.join(refsDir, `${id}.json`))
}

async function loadPaperTitle(id: string, papersDir?: string): Promise<string | null> {
  if (!papersDir) return null
  const p = await readJsonIfExists<PaperSchema>(path.join(papersDir, `${id}.json`))
  return p?.title ?? null
}

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

export async function selectFromRefs(
  seed: string,
  opts?: SelectFromRefsOptions
): Promise<SelectFromRefsResult> {
  const refsDir = opts?.refsDir ?? "data/cache/refs"
  const papersDir = opts?.papersDir ?? "data/cache/papers"

  // Load seed refs
  const seedRefs = await loadRefs(seed, refsDir)
  if (!seedRefs) {
    throw new Error(`Seed references not found: ${path.join(refsDir, `${seed}.json`)}`)
  }

  const seedId = seedRefs.sourceArxivId || seed
  const seedTitle = await loadPaperTitle(seedId, papersDir)

  // Build first-hop sources
  const sources: IdTitle[] = []
  for (const r of seedRefs.references ?? []) {
    if (!r?.arxivId) continue
    const title = r.title ?? (await loadPaperTitle(r.arxivId, papersDir))
    sources.push({ arxivId: r.arxivId, title: title ?? null })
  }

  // Eligible = first-hop refs that themselves have refs
  const eligible: Array<{ s: IdTitle; child: ReferenceSchema }> = []
  for (const s of sources) {
    const child = await loadRefs(s.arxivId, refsDir)
    if (child?.references?.length) {
      eligible.push({ s, child })
    }
  }

  if (eligible.length === 0) throw new Error(`No eligible first-hop sources with references found for seed ${seedId}.`);

  // RANDOM selectedSource (first hop)
  const picked = pickRandom(eligible)
  const selectedSource: IdTitle = picked.s

  // RANDOM target (second hop from selectedSource.refs)
  const twoHop = picked.child.references
  const r2 = pickRandom(twoHop)
  const target: IdTitle = {
    arxivId: r2.arxivId,
    title: (r2.title ?? (await loadPaperTitle(r2.arxivId, papersDir))) ?? null,
  }

  return {
    seed: { arxivId: seedId, title: seedTitle ?? null },
    sources,
    selectedSource,
    target,
  }
}