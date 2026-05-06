package store

import "context"

type RunSummary struct {
	Run      string
	LastTsMs int64
	LastStep int64
}

type Point struct {
	Step  int64
	TsMs  int64
	Value float64
}

type SqliteRun struct {
	Run          string
	ExperimentID string
	Status       string
	StartedAt    string
	EndedAt      *string
	GitHash      *string
	GitDirty     *int64
}

type SqliteExperiment struct {
	ID          string
	Name        string
	Project     string
	Description *string
	CreatedAt   string
}

type PollRequest struct {
	Run string

	ChartTags []string
	Cursors   map[string]int64

	InitialPoints   int
	MaxPointsPerTag int

	SummaryTags    []string
	PrefixesGPU    []string
	PrefixesRouter []string
}

type PollResponse struct {
	SeriesDelta   map[string][]Point
	SummaryLatest map[string]Point
	GPULatest     map[string]Point
	RouterLatest  map[string]Point
}

// MetricsStore is the interface for reading training metrics.
type MetricsStore interface {
	// NewestRun returns the most recent run by mtime - FAST, no DuckDB queries
	NewestRun(ctx context.Context) (string, error)
	// ListRuns returns runs sorted by recency - can be slow, call in background
	ListRuns(ctx context.Context, limit int) ([]RunSummary, error)
	// Poll fetches metrics for a run
	Poll(ctx context.Context, req PollRequest) (PollResponse, error)
}

type ExperimentsStore interface {
	ListExperiments(limit int) ([]SqliteExperiment, error)
	ListRuns(limit int) ([]SqliteRun, error)
	GetRun(run string) (*SqliteRun, error)
}
