package tui

import (
	"context"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/Noumena-Network/nmoe/tools/nmon/internal/store"
)

type pollKind int

const (
	pollFast pollKind = iota
	pollFull
)

// newestRunMsg - fast path, just mtime lookup, no DuckDB
type newestRunMsg struct {
	ID  int64
	Run string
	Dur time.Duration
	Err error
}

// runsLoadedMsg - background load of full run list
type runsLoadedMsg struct {
	ID    int64
	Limit int
	Runs  []store.RunSummary
	Dur   time.Duration
	Err   error
}

// runsProbedMsg - background probe of run max(step)/max(ts_ms) for a subset of runs
type runsProbedMsg struct {
	ID   int64
	Runs map[string]store.RunSummary
	Dur  time.Duration
	Err  error
}

// sqliteRunsLoadedMsg - background load of experiments
type sqliteRunsLoadedMsg struct {
	ID   int64
	Runs []store.SqliteRun
	Dur  time.Duration
	Err  error
}

// sqliteRunLoadedMsg - background load of a single run's experiments metadata
type sqliteRunLoadedMsg struct {
	Run string
	SR  *store.SqliteRun
	Dur time.Duration
	Err error
}

type pollResultMsg struct {
	ID   int64
	Kind pollKind
	At time.Time
	Dur time.Duration

	Run string

	SeriesDelta map[string][]store.Point

	GPULatest   map[string]store.Point
	RouterLatest map[string]store.Point

	SummaryLatest map[string]store.Point

	Err error
}

// cmdGetNewestRun - FAST path, just mtime, returns in <1s
func (a *App) cmdGetNewestRun() tea.Cmd {
	a.newestRunReqID++
	reqID := a.newestRunReqID
	a.newestRunLoading = true
	return func() tea.Msg {
		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		run, err := a.metrics.NewestRun(ctx)
		return newestRunMsg{ID: reqID, Run: run, Dur: time.Since(start), Err: err}
	}
}

// cmdLoadRuns - BACKGROUND, can take time, generous timeout
func (a *App) cmdLoadRuns(limit int) tea.Cmd {
	if limit <= 0 {
		limit = 50
	}
	a.runsReqID++
	reqID := a.runsReqID
	a.runsLoading = true
	return func() tea.Msg {
		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		runs, err := a.metrics.ListRuns(ctx, limit)
		return runsLoadedMsg{ID: reqID, Limit: limit, Runs: runs, Dur: time.Since(start), Err: err}
	}
}

// cmdLoadSqliteRuns - BACKGROUND, can take time, generous timeout
func (a *App) cmdLoadSqliteRuns(limit int) tea.Cmd {
	if limit <= 0 {
		limit = 200
	}
	a.sqliteRunsReqID++
	reqID := a.sqliteRunsReqID
	a.sqliteRunsLoading = true
	return func() tea.Msg {
		start := time.Now()
		runs, err := a.exp.ListRuns(limit)
		return sqliteRunsLoadedMsg{ID: reqID, Runs: runs, Dur: time.Since(start), Err: err}
	}
}

func (a *App) cmdLoadSqliteRun(run string) tea.Cmd {
	run = run
	return func() tea.Msg {
		start := time.Now()
		if run == "" {
			return sqliteRunLoadedMsg{Run: run, SR: nil, Dur: time.Since(start), Err: nil}
		}
		sr, err := a.exp.GetRun(run)
		return sqliteRunLoadedMsg{Run: run, SR: sr, Dur: time.Since(start), Err: err}
	}
}

func (a *App) cmdPoll(run string, cursors map[string]int64, kind pollKind) tea.Cmd {
	run = run
	maxPoints := a.opts.MaxPoints
	initialPoints := a.opts.InitialPoints

	timeout := 15 * time.Second
	charts := defaultCharts
	if kind == pollFast {
		timeout = 8 * time.Second
		if len(defaultCharts) >= 3 {
			charts = defaultCharts[:3]
		}
		if initialPoints <= 0 || initialPoints > 500 {
			initialPoints = 300
		}
		if maxPoints <= 0 || maxPoints > 1500 {
			maxPoints = 1000
		}
	}

	chartTags := make([]string, 0, len(charts))
	for _, c := range charts {
		chartTags = append(chartTags, c.Tag)
	}

	a.pollReqID++
	reqID := a.pollReqID
	a.pollInFlight = true
	return func() tea.Msg {
		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		summaryTags := []string{
			"train/loss",
			"throughput/tokens_per_s_gpu",
			"throughput/ms_per_step",
			"throughput/loader_wait_ms",
			"efficiency/tflops",
			"efficiency/fp8_tflops",
			"efficiency/bf16_tflops",
			"gpu_agg/mean_utilization_gpu",
			"gpu_agg/total_memory_used_gib",
			"gpu_agg/total_power_w",
			"gpu_agg/max_temperature_c",
			// RDEP/comm health (emitted rank-0 only when available)
			"comm/r0/capacity_utilization",
			"comm/r0/dropped_rows",
			"comm/r0/M_recv",
			"comm/r0/M_back",
			"comm/r0/capacity",
			// Timing health (when available)
			"time_ms/r0/fwd_total",
			"time_ms/r0/bwd_total",
			"time_ms/r0/opt_step",
			"time_ms/r0/zero2_reduce_scatter",
			"time_ms/r0/zero2_all_gather",
		}
		resp, err := a.metrics.Poll(ctx, store.PollRequest{
			Run:             run,
			ChartTags:       chartTags,
			Cursors:         cursors,
			InitialPoints:   initialPoints,
			MaxPointsPerTag: maxPoints,
			SummaryTags:     summaryTags,
			PrefixesGPU:     []string{"gpu/", "gpu_agg/"},
			PrefixesRouter:  []string{"router/layer_", "router_agg/"},
		})
		if err != nil {
			return pollResultMsg{ID: reqID, Kind: kind, At: time.Now(), Dur: time.Since(start), Run: run, Err: err}
		}

		return pollResultMsg{
			ID:           reqID,
			Kind:         kind,
			At:           time.Now(),
			Dur:          time.Since(start),
			Run:          run,
			SeriesDelta:  resp.SeriesDelta,
			GPULatest:    resp.GPULatest,
			RouterLatest: resp.RouterLatest,
			SummaryLatest: resp.SummaryLatest,
		}
	}
}

type runsProber interface {
	ProbeRuns(ctx context.Context, runs []string) (map[string]store.RunSummary, error)
}

func (a *App) cmdProbeRuns(runs []string) tea.Cmd {
	runs = append([]string(nil), runs...)
	a.runsProbeReqID++
	reqID := a.runsProbeReqID
	a.runsProbeInFlight = true
	return func() tea.Msg {
		p, ok := a.metrics.(runsProber)
		if !ok {
			return runsProbedMsg{ID: reqID, Runs: nil, Err: nil}
		}
		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		out, err := p.ProbeRuns(ctx, runs)
		return runsProbedMsg{ID: reqID, Runs: out, Dur: time.Since(start), Err: err}
	}
}
