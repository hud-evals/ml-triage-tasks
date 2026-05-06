import { describe, it, expect } from 'bun:test'
import { NextRequest } from 'next/server'
import { GET as runsHandler } from '@/app/api/runs/route'

import { mkdtempSync, rmSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import os from 'node:os'

import { DuckDBInstance } from '@duckdb/node-api'

describe('/api/runs contract (Parquet + SQLite merge)', () => {
  it('includes sqlite status and parquet last_step', async () => {
    const root = mkdtempSync(join(os.tmpdir(), 'nviz-fixture-'))
    const metricsDir = join(root, 'metrics')
    const expDb = join(root, 'experiments.db')

    process.env.NVIZ_METRICS_DIR = metricsDir
    process.env.NVIZ_EXPERIMENTS_DB = expDb

    const sqlLit = (s: string) => `'${s.replaceAll("'", "''")}'`

    // Create Parquet metrics for runA.
    const runA = 'runA'
    mkdirSync(join(metricsDir, runA), { recursive: true })
    {
      const inst = await DuckDBInstance.create(':memory:')
      const conn = await inst.connect()
      try {
        const now = Date.now()
        const p1 = join(metricsDir, runA, 'step_00000001.parquet')
        const p2 = join(metricsDir, runA, 'step_00000002.parquet')
        await conn.run(
          `COPY (SELECT ${sqlLit(runA)} AS run, 'train/loss' AS tag, 1::INTEGER AS step, ${now - 2000}::BIGINT AS ts_ms, 10.0::DOUBLE AS value) TO ${sqlLit(p1)} (FORMAT PARQUET)`
        )
        await conn.run(
          `COPY (SELECT ${sqlLit(runA)} AS run, 'train/loss' AS tag, 2::INTEGER AS step, ${now - 1000}::BIGINT AS ts_ms, 8.0::DOUBLE AS value) TO ${sqlLit(p2)} (FORMAT PARQUET)`
        )
      } finally {
        try { await conn.close() } catch {}
        try { await inst.close() } catch {}
      }
    }

    // Create SQLite experiments DB with runA + runB (no metrics).
    {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const { Database } = require('bun:sqlite')
      const db = new Database(expDb)
      try {
        db.exec(`
          CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            project TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
          );
        `)
        db.exec(`
          CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            config_json TEXT NOT NULL,
            git_hash TEXT,
            git_dirty INTEGER DEFAULT 0,
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            ended_at TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            results_json TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
          );
        `)
        db.run(`INSERT OR REPLACE INTO experiments(id,name,project,description,created_at) VALUES (?,?,?,?,datetime('now'))`, [
          'exp1','exp','proj','',
        ])
        db.run(`INSERT OR REPLACE INTO runs(id,experiment_id,config_json,git_hash,git_dirty,status,started_at) VALUES (?,?,?,?,?,?,datetime('now'))`, [
          runA, 'exp1', '{}', 'deadbeef', 0, 'running',
        ])
        db.run(`INSERT OR REPLACE INTO runs(id,experiment_id,config_json,git_hash,git_dirty,status,started_at) VALUES (?,?,?,?,?,?,datetime('now','-1 hour'))`, [
          'runB', 'exp1', '{}', 'deadbeef', 0, 'completed',
        ])
      } finally {
        try { db.close() } catch {}
      }
    }

    const url = new URL('http://localhost/api/runs')
    const req = new NextRequest(url)
    const res = await runsHandler(req as any)
    const body = await res.json() as any
    if (!res.ok) throw new Error(body?.error ? String(body.error) : JSON.stringify(body))
    const runs = body.runs as any[]
    expect(Array.isArray(runs)).toBe(true)

    const a = runs.find(r => r.run === runA)
    expect(a).toBeTruthy()
    expect(a.status).toBe('running')
    expect(a.last_step).toBe(2)

    const b = runs.find(r => r.run === 'runB')
    expect(b).toBeTruthy()
    expect(b.status).toBe('completed')
    expect(b.last_step).toBe(0)
    expect(b.last_ts).toBe(0)

    try { rmSync(root, { recursive: true, force: true }) } catch {}
  })
})
