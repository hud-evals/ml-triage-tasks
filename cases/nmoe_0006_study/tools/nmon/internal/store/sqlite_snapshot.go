package store

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	_ "modernc.org/sqlite"
)

type Experiments struct {
	dbPath string

	mu          sync.Mutex
	snapshotSig string
	snapshot    string
}

func NewExperiments(dbPath string) *Experiments { return &Experiments{dbPath: dbPath} }

func (e *Experiments) ListExperiments(limit int) ([]SqliteExperiment, error) {
	db, err := e.openSnapshot()
	if err != nil {
		return nil, err
	}
	if db == nil {
		return []SqliteExperiment{}, nil
	}
	defer db.Close()

	lim := clampInt(limit, 1, 2000)
	rows, err := db.Query(`
      SELECT id, name, project, description, created_at
      FROM experiments
      ORDER BY created_at DESC
      LIMIT ?
    `, lim)
	if err != nil {
		return nil, fmt.Errorf("experiments: list experiments: %w", err)
	}
	defer rows.Close()

	var out []SqliteExperiment
	for rows.Next() {
		var x SqliteExperiment
		if err := rows.Scan(&x.ID, &x.Name, &x.Project, &x.Description, &x.CreatedAt); err != nil {
			return nil, fmt.Errorf("experiments: scan experiment: %w", err)
		}
		out = append(out, x)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("experiments: rows: %w", err)
	}
	return out, nil
}

func (e *Experiments) ListRuns(limit int) ([]SqliteRun, error) {
	db, err := e.openSnapshot()
	if err != nil {
		return nil, err
	}
	if db == nil {
		return []SqliteRun{}, nil
	}
	defer db.Close()

	lim := clampInt(limit, 1, 2000)
	rows, err := db.Query(`
      SELECT
        id AS run,
        experiment_id,
        status,
        started_at,
        ended_at,
        git_hash,
        git_dirty
      FROM runs
      ORDER BY started_at DESC
      LIMIT ?
    `, lim)
	if err != nil {
		return nil, fmt.Errorf("experiments: list runs: %w", err)
	}
	defer rows.Close()

	var out []SqliteRun
	for rows.Next() {
		var x SqliteRun
		if err := rows.Scan(&x.Run, &x.ExperimentID, &x.Status, &x.StartedAt, &x.EndedAt, &x.GitHash, &x.GitDirty); err != nil {
			return nil, fmt.Errorf("experiments: scan run: %w", err)
		}
		out = append(out, x)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("experiments: rows: %w", err)
	}
	return out, nil
}

func (e *Experiments) GetRun(run string) (*SqliteRun, error) {
	db, err := e.openSnapshot()
	if err != nil {
		return nil, err
	}
	if db == nil {
		return nil, nil
	}
	defer db.Close()

	row := db.QueryRow(`
      SELECT
        id AS run,
        experiment_id,
        status,
        started_at,
        ended_at,
        git_hash,
        git_dirty
      FROM runs
      WHERE id = ?
    `, run)

	var x SqliteRun
	if err := row.Scan(&x.Run, &x.ExperimentID, &x.Status, &x.StartedAt, &x.EndedAt, &x.GitHash, &x.GitDirty); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("experiments: get run: %w", err)
	}
	return &x, nil
}

func (e *Experiments) openSnapshot() (*sql.DB, error) {
	path, err := e.ensureSnapshot()
	if err != nil {
		return nil, err
	}
	if path == "" {
		return nil, nil
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("experiments: open snapshot: %w", err)
	}
	return db, nil
}

func (e *Experiments) ensureSnapshot() (string, error) {
	src := filepath.Clean(e.dbPath)
	if src == "" {
		return "", nil
	}
	st, err := os.Stat(src)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("experiments: stat db: %w", err)
	}
	if !st.Mode().IsRegular() {
		return "", fmt.Errorf("experiments: db is not a file")
	}

	wal := src + "-wal"
	shm := src + "-shm"
	sig := fmt.Sprintf("%s:%d:%d", src, st.Size(), st.ModTime().UnixNano())
	if wst, err := os.Stat(wal); err == nil && wst.Mode().IsRegular() {
		sig += fmt.Sprintf(":%d:%d", wst.Size(), wst.ModTime().UnixNano())
	} else {
		sig += ":no-wal"
	}
	if sst, err := os.Stat(shm); err == nil && sst.Mode().IsRegular() {
		sig += fmt.Sprintf(":%d:%d", sst.Size(), sst.ModTime().UnixNano())
	} else {
		sig += ":no-shm"
	}

	e.mu.Lock()
	if e.snapshotSig == sig && e.snapshot != "" {
		path := e.snapshot
		e.mu.Unlock()
		return path, nil
	}
	e.mu.Unlock()

	dstDir := filepath.Join(os.TempDir(), "nmon_experiments")
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return "", fmt.Errorf("experiments: mkdir: %w", err)
	}
	dst := filepath.Join(dstDir, "experiments.db")

	if err := copyFile(src, dst); err != nil {
		return "", err
	}
	_ = copyFile(wal, dst+"-wal")
	_ = copyFile(shm, dst+"-shm")

	e.mu.Lock()
	e.snapshotSig = sig
	e.snapshot = dst
	e.mu.Unlock()
	return dst, nil
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func copyFile(src, dst string) error {
	st, err := os.Stat(src)
	if err != nil {
		return err
	}
	if !st.Mode().IsRegular() {
		return fmt.Errorf("experiments: not a regular file: %s", src)
	}
	in, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("experiments: open: %w", err)
	}
	defer in.Close()

	tmp := dst + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("experiments: create: %w", err)
	}
	if _, err := out.ReadFrom(in); err != nil {
		_ = out.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("experiments: copy: %w", err)
	}
	if err := out.Close(); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("experiments: close: %w", err)
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("experiments: rename: %w", err)
	}
	return nil
}
