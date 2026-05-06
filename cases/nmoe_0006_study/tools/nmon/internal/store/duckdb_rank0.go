package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	_ "github.com/marcboeker/go-duckdb"
)

var runNameRe = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

type DuckDBRank0 struct {
	metricsDir string

	mu     sync.Mutex
	run    string
	dbPath string
	db     *sql.DB
	conn   *sql.Conn
}

func NewDuckDBRank0(metricsDir string) *DuckDBRank0 { return &DuckDBRank0{metricsDir: metricsDir} }

func (s *DuckDBRank0) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closeLocked()
}

func (s *DuckDBRank0) closeLocked() error {
	var errs []error
	if s.conn != nil {
		errs = append(errs, s.conn.Close())
		s.conn = nil
	}
	if s.db != nil {
		errs = append(errs, s.db.Close())
		s.db = nil
	}
	s.run = ""
	s.dbPath = ""
	return errors.Join(errs...)
}

func (s *DuckDBRank0) ensureOpenLocked(ctx context.Context, run string) error {
	if run == "" {
		return fmt.Errorf("duckdb: empty run")
	}
	if !runNameRe.MatchString(run) {
		return fmt.Errorf("duckdb: invalid run %q", run)
	}
	runDir, err := s.rank0DBPath(run)
	if err != nil {
		return err
	}
	if s.conn != nil && s.run == run && s.dbPath == runDir {
		return nil
	}

	_ = s.closeLocked()

	db, err := sql.Open("duckdb", "")
	if err != nil {
		return fmt.Errorf("duckdb: open in-memory: %w", err)
	}
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)

	conn, err := db.Conn(ctx)
	if err != nil {
		_ = db.Close()
		return fmt.Errorf("duckdb: conn: %w", err)
	}

	if _, err := conn.ExecContext(ctx, "SET enable_progress_bar=false"); err != nil {
		_ = conn.Close()
		_ = db.Close()
		return fmt.Errorf("duckdb: set options: %w", err)
	}
	// Create view over parquet files (no locking, concurrent reads supported).
	// Backward-compat: if no parquet exists, fall back to rank_0.duckdb for older runs.
	parquetGlob := filepath.Join(runDir, "step_*.parquet")
	parquetFiles, _ := filepath.Glob(parquetGlob)
	if len(parquetFiles) > 0 {
		if _, err := conn.ExecContext(
			ctx,
			fmt.Sprintf("CREATE OR REPLACE VIEW r0_metrics AS SELECT * FROM read_parquet(%s, union_by_name=true, filename=false)", sqlLit(parquetGlob)),
		); err != nil {
			_ = conn.Close()
			_ = db.Close()
			return fmt.Errorf("duckdb: create parquet view: %w", err)
		}
	} else {
		dbPath := filepath.Join(runDir, "rank_0.duckdb")
		if _, err := os.Stat(dbPath); err != nil {
			_ = conn.Close()
			_ = db.Close()
			return fmt.Errorf("duckdb: no parquet or rank_0.duckdb for run %q", run)
		}
		if _, err := conn.ExecContext(ctx, fmt.Sprintf("ATTACH %s AS r0 (READ_ONLY);", sqlLit(dbPath))); err != nil {
			_ = conn.Close()
			_ = db.Close()
			return fmt.Errorf("duckdb: attach rank_0: %w", err)
		}
		if _, err := conn.ExecContext(ctx, "CREATE OR REPLACE VIEW r0_metrics AS SELECT * FROM r0.metrics"); err != nil {
			_ = conn.Close()
			_ = db.Close()
			return fmt.Errorf("duckdb: create duckdb view: %w", err)
		}
	}

	s.db = db
	s.conn = conn
	s.run = run
	s.dbPath = runDir
	return nil
}

func (s *DuckDBRank0) rank0DBPath(run string) (string, error) {
	root := filepath.Clean(s.metricsDir)
	if root == "" {
		return "", fmt.Errorf("duckdb: empty metrics dir")
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("duckdb: abs metrics dir: %w", err)
	}
	// Run directory: {metrics_dir}/{run}
	target := filepath.Join(absRoot, run)
	target = filepath.Clean(target)
	if target != absRoot && !strings.HasPrefix(target, absRoot+string(filepath.Separator)) {
		return "", fmt.Errorf("duckdb: invalid run path")
	}
	return target, nil
}

func sqlLit(s string) string { return "'" + strings.ReplaceAll(s, "'", "''") + "'" }

// NewestRun returns the most recent run by mtime - FAST, no DuckDB queries
func (s *DuckDBRank0) NewestRun(ctx context.Context) (string, error) {
	root := s.metricsDir
	ents, err := os.ReadDir(root)
	if err != nil {
		return "", fmt.Errorf("duckdb: read metrics dir: %w", err)
	}

	var newest string
	var newestMtime int64
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if !runNameRe.MatchString(name) {
			continue
		}
		runDir := filepath.Join(root, name)
		parquetFiles, _ := filepath.Glob(filepath.Join(runDir, "step_*.parquet"))
		var mtime int64
		if len(parquetFiles) > 0 {
			// Use mtime of newest parquet file
			for _, pf := range parquetFiles {
				if info, err := os.Stat(pf); err == nil {
					if t := info.ModTime().UnixMilli(); t > mtime {
						mtime = t
					}
				}
			}
		} else if info, err := os.Stat(filepath.Join(runDir, "rank_0.duckdb")); err == nil {
			// Backward-compat (older runs): use duckdb mtime.
			mtime = info.ModTime().UnixMilli()
		} else {
			continue
		}
		if mtime > newestMtime {
			newestMtime = mtime
			newest = name
		}
	}
	return newest, nil
}

func (s *DuckDBRank0) ListRuns(ctx context.Context, limit int) ([]RunSummary, error) {
	if limit <= 0 {
		limit = 50
	}
	root := s.metricsDir
	ents, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("duckdb: read metrics dir: %w", err)
	}

	type candidate struct {
		run    string
		runDir string
		mtime  int64
	}
	var cands []candidate
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if !runNameRe.MatchString(name) {
			continue
		}
		runDir, err := s.rank0DBPath(name)
		if err != nil {
			continue
		}
		var mtime int64
		parquetFiles, _ := filepath.Glob(filepath.Join(runDir, "step_*.parquet"))
		if len(parquetFiles) > 0 {
			for _, pf := range parquetFiles {
				if info, err := os.Stat(pf); err == nil {
					if t := info.ModTime().UnixMilli(); t > mtime {
						mtime = t
					}
				}
			}
		} else if info, err := os.Stat(filepath.Join(runDir, "rank_0.duckdb")); err == nil {
			mtime = info.ModTime().UnixMilli()
		} else {
			continue
		}
		cands = append(cands, candidate{run: name, runDir: runDir, mtime: mtime})
	}

	sort.Slice(cands, func(i, j int) bool {
		if cands[i].mtime != cands[j].mtime {
			return cands[i].mtime > cands[j].mtime
		}
		return cands[i].run < cands[j].run
	})
	if len(cands) > limit {
		cands = cands[:limit]
	}

	out := make([]RunSummary, 0, len(cands))
	for _, c := range cands {
		lastTs, lastStep := int64(0), int64(0)
		db, err := sql.Open("duckdb", "")
		if err != nil {
			continue
		}
		db.SetMaxOpenConns(1)
		conn, err := db.Conn(ctx)
		if err != nil {
			_ = db.Close()
			continue
		}
		parquetGlob := filepath.Join(c.runDir, "step_*.parquet")
		parquetFiles, _ := filepath.Glob(parquetGlob)
		if len(parquetFiles) > 0 {
			_, err = conn.ExecContext(ctx, fmt.Sprintf("CREATE OR REPLACE VIEW r0_metrics AS SELECT * FROM read_parquet(%s, union_by_name=true, filename=false)", sqlLit(parquetGlob)))
		} else {
			dbPath := filepath.Join(c.runDir, "rank_0.duckdb")
			_, err = conn.ExecContext(ctx, fmt.Sprintf("ATTACH %s AS r0 (READ_ONLY);", sqlLit(dbPath)))
			if err == nil {
				_, err = conn.ExecContext(ctx, "CREATE OR REPLACE VIEW r0_metrics AS SELECT * FROM r0.metrics")
			}
		}
		if err == nil {
			row := conn.QueryRowContext(ctx, "SELECT max(ts_ms) AS last_ts, max(step) AS last_step FROM r0_metrics")
			_ = row.Scan(&lastTs, &lastStep)
		}
		_ = conn.Close()
		_ = db.Close()
		out = append(out, RunSummary{Run: c.run, LastTsMs: lastTs, LastStep: lastStep})
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].LastTsMs != out[j].LastTsMs {
			return out[i].LastTsMs > out[j].LastTsMs
		}
		if out[i].LastStep != out[j].LastStep {
			return out[i].LastStep > out[j].LastStep
		}
		return out[i].Run < out[j].Run
	})
	return out, nil
}

func (s *DuckDBRank0) Poll(ctx context.Context, req PollRequest) (PollResponse, error) {
	run := req.Run
	if run == "" {
		return PollResponse{}, fmt.Errorf("duckdb: empty run")
	}

	initialPoints := req.InitialPoints
	if initialPoints <= 0 {
		initialPoints = 2000
	}
	maxPointsPerTag := req.MaxPointsPerTag
	if maxPointsPerTag <= 0 {
		maxPointsPerTag = 4000
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureOpenLocked(ctx, run); err != nil {
		return PollResponse{}, err
	}
	conn := s.conn
	if conn == nil {
		return PollResponse{}, fmt.Errorf("duckdb: not open")
	}

	delta := make(map[string][]Point, len(req.ChartTags))
	for _, tag := range req.ChartTags {
		after, ok := req.Cursors[tag]
		if !ok {
			pts, err := tailLocked(ctx, conn, run, tag, initialPoints)
			if err != nil {
				return PollResponse{}, err
			}
			delta[tag] = pts
			continue
		}
		pts, err := sinceLocked(ctx, conn, run, tag, after, maxPointsPerTag)
		if err != nil {
			return PollResponse{}, err
		}
		if len(pts) > 0 {
			delta[tag] = pts
		}
	}

	summaryLatest, err := latestForTagsLocked(ctx, conn, run, req.SummaryTags)
	if err != nil {
		return PollResponse{}, err
	}
	gpuLatest, err := latestForPrefixesLocked(ctx, conn, run, req.PrefixesGPU)
	if err != nil {
		return PollResponse{}, err
	}
	routerLatest, err := latestForPrefixesLocked(ctx, conn, run, req.PrefixesRouter)
	if err != nil {
		return PollResponse{}, err
	}

	return PollResponse{
		SeriesDelta:   delta,
		SummaryLatest: summaryLatest,
		GPULatest:     gpuLatest,
		RouterLatest:  routerLatest,
	}, nil
}

// ProbeRuns returns max(step)/max(ts_ms) for the provided runs.
//
// This is used by the TUI to lazily populate the experiments list without
// paying a full ListRuns() probe cost for large metrics directories.
func (s *DuckDBRank0) ProbeRuns(ctx context.Context, runs []string) (map[string]RunSummary, error) {
	out := make(map[string]RunSummary, len(runs))
	for _, run := range runs {
		if run == "" || !runNameRe.MatchString(run) {
			continue
		}
		runDir, err := s.rank0DBPath(run)
		if err != nil {
			continue
		}
		lastTs, lastStep := int64(0), int64(-1)
		parquetGlob := filepath.Join(runDir, "step_*.parquet")
		parquetFiles, _ := filepath.Glob(parquetGlob)
		if len(parquetFiles) > 0 {
			for _, pf := range parquetFiles {
				if info, err := os.Stat(pf); err == nil {
					if t := info.ModTime().UnixMilli(); t > lastTs {
						lastTs = t
					}
				}
			}
		} else if info, err := os.Stat(filepath.Join(runDir, "rank_0.duckdb")); err == nil {
			lastTs = info.ModTime().UnixMilli()
		} else {
			continue
		}

		db, err := sql.Open("duckdb", "")
		if err != nil {
			out[run] = RunSummary{Run: run, LastTsMs: lastTs, LastStep: lastStep}
			continue
		}
		db.SetMaxOpenConns(1)
		conn, err := db.Conn(ctx)
		if err != nil {
			_ = db.Close()
			out[run] = RunSummary{Run: run, LastTsMs: lastTs, LastStep: lastStep}
			continue
		}

		err = nil
		if len(parquetFiles) > 0 {
			_, err = conn.ExecContext(ctx, fmt.Sprintf("CREATE OR REPLACE VIEW r0_metrics AS SELECT * FROM read_parquet(%s, union_by_name=true, filename=false)", sqlLit(parquetGlob)))
		} else {
			dbPath := filepath.Join(runDir, "rank_0.duckdb")
			_, err = conn.ExecContext(ctx, fmt.Sprintf("ATTACH %s AS r0 (READ_ONLY);", sqlLit(dbPath)))
			if err == nil {
				_, err = conn.ExecContext(ctx, "CREATE OR REPLACE VIEW r0_metrics AS SELECT * FROM r0.metrics")
			}
		}
		if err == nil {
			row := conn.QueryRowContext(ctx, "SELECT max(ts_ms), max(step) FROM r0_metrics WHERE run = ?", run)
			_ = row.Scan(&lastTs, &lastStep)
		}
		_ = conn.Close()
		_ = db.Close()

		out[run] = RunSummary{Run: run, LastTsMs: lastTs, LastStep: lastStep}
	}
	return out, nil
}

func tailLocked(ctx context.Context, conn *sql.Conn, run, tag string, limit int) ([]Point, error) {
	if limit <= 0 {
		return nil, nil
	}
	rows, err := conn.QueryContext(
		ctx,
		"SELECT step, ts_ms, value FROM r0_metrics WHERE run = ? AND tag = ? ORDER BY step DESC LIMIT ?",
		run, tag, limit,
	)
	if err != nil {
		return nil, fmt.Errorf("duckdb: tail query: %w", err)
	}
	defer rows.Close()

	var out []Point
	for rows.Next() {
		var p Point
		if err := rows.Scan(&p.Step, &p.TsMs, &p.Value); err != nil {
			return nil, fmt.Errorf("duckdb: tail scan: %w", err)
		}
		out = append(out, p)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("duckdb: tail rows: %w", err)
	}
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out, nil
}

func sinceLocked(ctx context.Context, conn *sql.Conn, run, tag string, afterStep int64, limit int) ([]Point, error) {
	if limit <= 0 {
		return nil, nil
	}
	rows, err := conn.QueryContext(
		ctx,
		"SELECT step, ts_ms, value FROM r0_metrics WHERE run = ? AND tag = ? AND step > ? ORDER BY step ASC LIMIT ?",
		run, tag, afterStep, limit,
	)
	if err != nil {
		return nil, fmt.Errorf("duckdb: since query: %w", err)
	}
	defer rows.Close()

	var out []Point
	for rows.Next() {
		var p Point
		if err := rows.Scan(&p.Step, &p.TsMs, &p.Value); err != nil {
			return nil, fmt.Errorf("duckdb: since scan: %w", err)
		}
		out = append(out, p)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("duckdb: since rows: %w", err)
	}
	return out, nil
}

func latestForTagsLocked(ctx context.Context, conn *sql.Conn, run string, tags []string) (map[string]Point, error) {
	if len(tags) == 0 {
		return map[string]Point{}, nil
	}

	var b strings.Builder
	b.WriteString("WITH u AS (SELECT tag, step, ts_ms, value FROM r0_metrics WHERE run = ? AND tag IN (")
	args := make([]any, 0, 1+len(tags))
	args = append(args, run)
	for i := range tags {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString("?")
		args = append(args, tags[i])
	}
	b.WriteString(")) SELECT tag, step, ts_ms, value FROM u QUALIFY row_number() OVER (PARTITION BY tag ORDER BY ts_ms DESC, step DESC) = 1")

	rows, err := conn.QueryContext(ctx, b.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("duckdb: latest tags query: %w", err)
	}
	defer rows.Close()

	out := make(map[string]Point, len(tags))
	for rows.Next() {
		var tag string
		var p Point
		if err := rows.Scan(&tag, &p.Step, &p.TsMs, &p.Value); err != nil {
			return nil, fmt.Errorf("duckdb: latest tags scan: %w", err)
		}
		out[tag] = p
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("duckdb: latest tags rows: %w", err)
	}
	return out, nil
}

func latestForPrefixesLocked(ctx context.Context, conn *sql.Conn, run string, prefixes []string) (map[string]Point, error) {
	if len(prefixes) == 0 {
		return map[string]Point{}, nil
	}

	var where strings.Builder
	args := make([]any, 0, 1+len(prefixes))
	args = append(args, run)
	for i := range prefixes {
		if i > 0 {
			where.WriteString(" OR ")
		}
		where.WriteString("tag LIKE ?")
		args = append(args, prefixes[i]+"%")
	}

	sql := fmt.Sprintf(`
      WITH u AS (
        SELECT tag, step, ts_ms, value
        FROM r0_metrics
        WHERE run = ? AND (%s)
      )
      SELECT tag, step, ts_ms, value
      FROM u
      QUALIFY row_number() OVER (PARTITION BY tag ORDER BY ts_ms DESC, step DESC) = 1
    `, where.String())

	rows, err := conn.QueryContext(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("duckdb: latest prefixes query: %w", err)
	}
	defer rows.Close()

	out := make(map[string]Point, 64)
	for rows.Next() {
		var tag string
		var p Point
		if err := rows.Scan(&tag, &p.Step, &p.TsMs, &p.Value); err != nil {
			return nil, fmt.Errorf("duckdb: latest prefixes scan: %w", err)
		}
		out[tag] = p
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("duckdb: latest prefixes rows: %w", err)
	}
	return out, nil
}

func (s *DuckDBRank0) Tail(ctx context.Context, run, tag string, limit int) ([]Point, error) {
	if limit <= 0 {
		return nil, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureOpenLocked(ctx, run); err != nil {
		return nil, err
	}
	conn := s.conn

	rows, err := conn.QueryContext(
		ctx,
		"SELECT step, ts_ms, value FROM r0_metrics WHERE run = ? AND tag = ? ORDER BY step DESC LIMIT ?",
		run, tag, limit,
	)
	if err != nil {
		return nil, fmt.Errorf("duckdb: tail query: %w", err)
	}
	defer rows.Close()

	var out []Point
	for rows.Next() {
		var p Point
		if err := rows.Scan(&p.Step, &p.TsMs, &p.Value); err != nil {
			return nil, fmt.Errorf("duckdb: tail scan: %w", err)
		}
		out = append(out, p)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("duckdb: tail rows: %w", err)
	}

	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out, nil
}

func (s *DuckDBRank0) Since(ctx context.Context, run, tag string, afterStep int64, limit int) ([]Point, error) {
	if limit <= 0 {
		return nil, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureOpenLocked(ctx, run); err != nil {
		return nil, err
	}
	conn := s.conn

	rows, err := conn.QueryContext(
		ctx,
		"SELECT step, ts_ms, value FROM r0_metrics WHERE run = ? AND tag = ? AND step > ? ORDER BY step ASC LIMIT ?",
		run, tag, afterStep, limit,
	)
	if err != nil {
		return nil, fmt.Errorf("duckdb: since query: %w", err)
	}
	defer rows.Close()

	var out []Point
	for rows.Next() {
		var p Point
		if err := rows.Scan(&p.Step, &p.TsMs, &p.Value); err != nil {
			return nil, fmt.Errorf("duckdb: since scan: %w", err)
		}
		out = append(out, p)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("duckdb: since rows: %w", err)
	}
	return out, nil
}

func (s *DuckDBRank0) LatestForTags(ctx context.Context, run string, tags []string) (map[string]Point, error) {
	if len(tags) == 0 {
		return map[string]Point{}, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureOpenLocked(ctx, run); err != nil {
		return nil, err
	}
	conn := s.conn

	var b strings.Builder
	b.WriteString("WITH u AS (SELECT tag, step, ts_ms, value FROM r0_metrics WHERE run = ? AND tag IN (")
	args := make([]any, 0, 1+len(tags))
	args = append(args, run)
	for i := range tags {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString("?")
		args = append(args, tags[i])
	}
	b.WriteString(")) SELECT tag, step, ts_ms, value FROM u QUALIFY row_number() OVER (PARTITION BY tag ORDER BY ts_ms DESC, step DESC) = 1")

	rows, err := conn.QueryContext(ctx, b.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("duckdb: latest tags query: %w", err)
	}
	defer rows.Close()

	out := make(map[string]Point, len(tags))
	for rows.Next() {
		var tag string
		var p Point
		if err := rows.Scan(&tag, &p.Step, &p.TsMs, &p.Value); err != nil {
			return nil, fmt.Errorf("duckdb: latest tags scan: %w", err)
		}
		out[tag] = p
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("duckdb: latest tags rows: %w", err)
	}
	return out, nil
}

func (s *DuckDBRank0) LatestForPrefixes(ctx context.Context, run string, prefixes []string) (map[string]Point, error) {
	if len(prefixes) == 0 {
		return map[string]Point{}, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureOpenLocked(ctx, run); err != nil {
		return nil, err
	}
	conn := s.conn

	var where strings.Builder
	args := make([]any, 0, 1+len(prefixes))
	args = append(args, run)
	for i := range prefixes {
		if i > 0 {
			where.WriteString(" OR ")
		}
		where.WriteString("tag LIKE ?")
		args = append(args, prefixes[i]+"%")
	}

	sql := fmt.Sprintf(`
      WITH u AS (
        SELECT tag, step, ts_ms, value
        FROM r0_metrics
        WHERE run = ? AND (%s)
      )
      SELECT tag, step, ts_ms, value
      FROM u
      QUALIFY row_number() OVER (PARTITION BY tag ORDER BY ts_ms DESC, step DESC) = 1
    `, where.String())

	rows, err := conn.QueryContext(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("duckdb: latest prefixes query: %w", err)
	}
	defer rows.Close()

	out := make(map[string]Point, 64)
	for rows.Next() {
		var tag string
		var p Point
		if err := rows.Scan(&tag, &p.Step, &p.TsMs, &p.Value); err != nil {
			return nil, fmt.Errorf("duckdb: latest prefixes scan: %w", err)
		}
		out[tag] = p
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("duckdb: latest prefixes rows: %w", err)
	}
	return out, nil
}
