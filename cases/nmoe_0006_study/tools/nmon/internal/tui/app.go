package tui

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/Noumena-Network/nmoe/tools/nmon/internal/store"
	"github.com/Noumena-Network/nmoe/tools/nmon/internal/widgets"
)

type viewKind int

const (
	viewDashboard viewKind = iota
	viewExperiments
	viewLeaderboard
)

type tickMsg time.Time
type refreshNowMsg struct{}

type App struct {
	opts Options

	width  int
	height int

	view viewKind

	currentRun string

	runs       []store.RunSummary
	sqliteRuns map[string]store.SqliteRun
	runsCursor int
	runsLimit  int

	runStatsCache map[string]store.RunSummary

	runsProbeReqID    int64
	runsProbeInFlight bool

	newestRunReqID    int64
	runsReqID         int64
	sqliteRunsReqID   int64
	pollReqID         int64
	newestRunLoading  bool
	runsLoading       bool
	sqliteRunsLoading bool
	pollInFlight      bool

	dashCards     []dashCardCache
	dashMainW     int
	dashMainH     int
	dashMain      string
	dashMainDirty bool

	rightW     int
	rightH     int
	right      string
	rightDirty bool

	gpuParsed []gpuInfo
	gpuTable  []widgets.GpuInfo

	routerLayers []routerLayer
	routerAgg    []string

	series  map[string][]store.Point
	cursors map[string]int64

	gpuLatest     map[string]store.Point
	routerLatest  map[string]store.Point
	summaryLatest map[string]store.Point

	sqliteRun *store.SqliteRun

	lastPoll   time.Time
	lastUpdate time.Time

	lastNewestRunDur  time.Duration
	lastListRunsDur   time.Duration
	lastProbeRunsDur  time.Duration
	lastSqliteRunsDur time.Duration
	lastSqliteRunDur  time.Duration
	lastPollDur       time.Duration

	err error

	metrics store.MetricsStore
	exp     store.ExperimentsStore

	leaderboard       []LeaderboardEntry
	leaderboardLoaded bool

	showHelp bool

	zoomActive bool
	zoomCard   int
	zoomN      int
}

func NewApp(opts Options) *App {
	return &App{
		opts:          opts,
		view:          viewDashboard,
		sqliteRuns:    make(map[string]store.SqliteRun),
		runStatsCache: make(map[string]store.RunSummary),
		series:        make(map[string][]store.Point),
		cursors:       make(map[string]int64),
		runsLimit:     50,
		dashCards:     newDashCards(),
		dashMainDirty: true,
		rightDirty:    true,
	}
}

func (a *App) Init() tea.Cmd {
	// Initialize stores (local DuckDB/SQLite or remote k8s exec)
	if a.opts.K8sEnabled {
		execStore, err := store.NewK8sExecStore(store.K8sExecOptions{
			Namespace:   a.opts.K8sNamespace,
			PodName:     a.opts.K8sPodName,
			PodSelector: a.opts.K8sPodSelector,
			Container:   a.opts.K8sContainer,
			MetricsDir:  a.opts.MetricsDir,
			Kubeconfig:  a.opts.K8sKubeconfig,
		})
		if err != nil {
			a.metrics = store.NewErrMetricsStore(err)
			a.exp = store.NewErrExperimentsStore(err)
		} else {
			a.metrics = execStore
			a.exp = store.NewK8sExperimentsStoreFromExec(execStore, a.opts.ExperimentsDB)
		}
	} else {
		a.metrics = store.NewDuckDBRank0(a.opts.MetricsDir)
		a.exp = store.NewExperiments(a.opts.ExperimentsDB)
	}

	// Fast render: pick the run and start polling; load multi-run inventory on demand (experiments view).
	if a.opts.RunID != "" {
		a.selectRun(a.opts.RunID)
		return tea.Batch(
			a.cmdLoadSqliteRun(a.currentRun),
			a.cmdPoll(a.currentRun, copyCursors(a.cursors), pollFast),
			a.tickCmd(),
		)
	}
	return tea.Batch(
		a.cmdGetNewestRun(),
		a.tickCmd(),
	)
}

func (a *App) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch m := msg.(type) {
	case tea.WindowSizeMsg:
		a.width, a.height = m.Width, m.Height
		a.markAllDashCardsDirty()
		a.markRightSidebarDirty()
		a.rebuildRightSidebarCache()
		if a.view == viewExperiments {
			return a, a.maybeProbeVisibleRuns()
		}
		return a, nil

	case tea.KeyMsg:
		switch m.String() {
		case "ctrl+c":
			return a, tea.Quit
		case "q":
			return a, tea.Quit
		case "?", "h":
			a.showHelp = !a.showHelp
			return a, nil
		}
		if a.showHelp {
			switch m.String() {
			case "esc", "enter":
				a.showHelp = false
				return a, nil
			default:
				return a, nil
			}
		}
		if a.zoomActive {
			return a.updateZoomKeys(m)
		}
		if a.view == viewExperiments {
			return a.updateExperimentsKeys(m)
		}
		if a.view == viewLeaderboard {
			return a.updateLeaderboardKeys(m)
		}
		switch m.String() {
		case "q":
			return a, tea.Quit
		case "d":
			a.view = viewDashboard
			return a, nil
		case "e":
			a.view = viewExperiments
			var cmds []tea.Cmd
			if len(a.runs) == 0 && !a.runsLoading {
				cmds = append(cmds, a.cmdLoadRuns(a.runsLimit))
			}
			if len(a.sqliteRuns) == 0 && !a.sqliteRunsLoading {
				cmds = append(cmds, a.cmdLoadSqliteRuns(200))
			}
			cmds = append(cmds, a.maybeProbeVisibleRuns())
			if len(cmds) == 0 {
				return a, nil
			}
			return a, tea.Batch(cmds...)
		case "l":
			a.view = viewLeaderboard
			if !a.leaderboardLoaded {
				a.loadLeaderboard()
			}
			return a, nil
		case "r":
			return a, func() tea.Msg { return refreshNowMsg{} }
		case "z":
			if a.view == viewDashboard && len(a.dashCards) > 0 {
				a.zoomActive = true
				a.zoomCard = 0
				a.zoomN = 2000
				return a, nil
			}
		}

	case newestRunMsg:
		if m.ID != a.newestRunReqID {
			return a, nil
		}
		a.newestRunLoading = false
		a.lastNewestRunDur = m.Dur
		if m.Err != nil {
			a.err = m.Err
			if len(a.runs) == 0 && !a.runsLoading {
				return a, tea.Batch(
					a.cmdLoadRuns(a.runsLimit),
					a.cmdLoadSqliteRuns(200),
				)
			}
			return a, nil
		}
		if m.Run != "" && a.currentRun == "" && !a.pollInFlight {
			a.selectRun(m.Run)
			return a, tea.Batch(
				a.cmdLoadSqliteRun(a.currentRun),
				a.cmdPoll(a.currentRun, copyCursors(a.cursors), pollFast),
			)
		}
		if a.currentRun == "" && len(a.runs) == 0 && !a.runsLoading {
			return a, tea.Batch(
				a.cmdLoadRuns(a.runsLimit),
				a.cmdLoadSqliteRuns(200),
			)
		}
		return a, nil

	case runsLoadedMsg:
		// BACKGROUND: full run list loaded (may take time over k8s exec)
		if m.ID != a.runsReqID {
			return a, nil
		}
		a.runsLoading = false
		a.lastListRunsDur = m.Dur
		if m.Limit > 0 {
			a.runsLimit = m.Limit
		}
		if m.Err != nil {
			a.err = m.Err
			return a, nil
		}
		a.runs = m.Runs
		for i := range a.runs {
			if cached, ok := a.runStatsCache[a.runs[i].Run]; ok && cached.LastStep >= 0 {
				a.runs[i].LastStep = cached.LastStep
				if cached.LastTsMs > 0 {
					a.runs[i].LastTsMs = cached.LastTsMs
				}
			}
		}
		if len(a.runs) == 0 {
			a.runsCursor = 0
		} else if a.runsCursor >= len(a.runs) {
			a.runsCursor = len(a.runs) - 1
		}
		// Only auto-select if we don't already have a run (newestRunMsg may have already set one)
		if a.currentRun == "" && len(a.runs) > 0 && !a.pollInFlight {
			a.selectRun(a.runs[0].Run)
			return a, tea.Batch(
				a.cmdLoadSqliteRun(a.currentRun),
				a.cmdPoll(a.currentRun, copyCursors(a.cursors), pollFast),
			)
		}
		if a.view == viewExperiments {
			return a, a.maybeProbeVisibleRuns()
		}
		return a, nil

	case sqliteRunsLoadedMsg:
		if m.ID != a.sqliteRunsReqID {
			return a, nil
		}
		a.sqliteRunsLoading = false
		a.lastSqliteRunsDur = m.Dur
		if m.Err != nil {
			a.err = m.Err
			return a, nil
		}
		a.sqliteRuns = make(map[string]store.SqliteRun, len(m.Runs))
		for _, r := range m.Runs {
			a.sqliteRuns[r.Run] = r
		}
		if a.view == viewExperiments {
			return a, a.maybeProbeVisibleRuns()
		}
		return a, nil

	case sqliteRunLoadedMsg:
		if m.Run == "" || m.Run != a.currentRun {
			return a, nil
		}
		a.lastSqliteRunDur = m.Dur
		if m.Err != nil {
			// Non-fatal: run metadata is optional for the dashboard.
			return a, nil
		}
		a.sqliteRun = m.SR
		return a, nil

	case runsProbedMsg:
		if m.ID != a.runsProbeReqID {
			return a, nil
		}
		a.runsProbeInFlight = false
		a.lastProbeRunsDur = m.Dur
		if m.Err != nil {
			return a, nil
		}
		for run, rs := range m.Runs {
			if rs.Run == "" {
				rs.Run = run
			}
			if rs.LastStep < 0 {
				continue
			}
			a.runStatsCache[run] = rs
		}
		for i := range a.runs {
			if cached, ok := a.runStatsCache[a.runs[i].Run]; ok && cached.LastStep >= 0 {
				a.runs[i].LastStep = cached.LastStep
				if cached.LastTsMs > 0 {
					a.runs[i].LastTsMs = cached.LastTsMs
				}
			}
		}
		return a, nil

	case refreshNowMsg:
		if a.currentRun == "" {
			if !a.newestRunLoading {
				return a, a.cmdGetNewestRun()
			}
			if !a.runsLoading {
				return a, tea.Batch(
					a.cmdLoadRuns(a.runsLimit),
					a.cmdLoadSqliteRuns(200),
				)
			}
			return a, nil
		}
		return a, a.cmdPoll(a.currentRun, copyCursors(a.cursors), pollFull)

	case tickMsg:
		var cmds []tea.Cmd
		if a.currentRun != "" && !a.pollInFlight {
			cmds = append(cmds, a.cmdPoll(a.currentRun, copyCursors(a.cursors), pollFull))
		} else if a.currentRun == "" && !a.newestRunLoading {
			cmds = append(cmds, a.cmdGetNewestRun())
		}
		cmds = append(cmds, a.tickCmd())
		return a, tea.Batch(cmds...)

	case pollResultMsg:
		if m.ID != a.pollReqID {
			return a, nil
		}
		a.pollInFlight = false
		a.lastPollDur = m.Dur
		a.lastPoll = m.At
		if m.Run != "" && m.Run != a.currentRun {
			return a, nil
		}
		if m.Err != nil {
			a.err = m.Err
			return a, nil
		}
		a.err = nil
		a.lastUpdate = m.At
		a.gpuLatest = m.GPULatest
		a.routerLatest = m.RouterLatest
		a.summaryLatest = m.SummaryLatest

		a.gpuParsed = parseGpus(a.gpuLatest)
		a.gpuTable = make([]widgets.GpuInfo, 0, len(a.gpuParsed))
		for _, g := range a.gpuParsed {
			util := g.Util
			if util <= 1 && util > 0 {
				util *= 100
			}
			fan := g.Fan
			if fan <= 1 && fan > 0 {
				fan *= 100
			}
			a.gpuTable = append(a.gpuTable, widgets.GpuInfo{
				Index: g.Index,
				Temp:  g.Temp,
				Fan:   fan,
				Power: g.Pwr,
				PwrL:  g.PwrL,
				MemU:  g.MemU,
				MemT:  g.MemT,
				Util:  util,
			})
		}
		a.routerLayers, a.routerAgg = parseRouter(a.routerLatest)
		a.markRightSidebarDirty()
		a.rebuildRightSidebarCache()

		for tag, pts := range m.SeriesDelta {
			if len(pts) == 0 {
				continue
			}
			if _, ok := a.cursors[tag]; !ok {
				a.series[tag] = append([]store.Point(nil), pts...)
			} else {
				a.series[tag] = append(a.series[tag], pts...)
			}
			if maxN := a.opts.MaxPoints; maxN > 0 && len(a.series[tag]) > maxN {
				a.series[tag] = a.series[tag][len(a.series[tag])-maxN:]
			}
			a.cursors[tag] = a.series[tag][len(a.series[tag])-1].Step
			a.markDashCardDirty(tag)
		}
		if m.Kind == pollFast {
			if !a.pollInFlight {
				return a, a.cmdPoll(a.currentRun, copyCursors(a.cursors), pollFull)
			}
		}
		return a, nil
	}

	return a, nil
}

func (a *App) View() string {
	if a.width <= 0 || a.height <= 0 {
		return ""
	}

	if a.width < 79 {
		msg := lipgloss.NewStyle().Padding(1, 2).Render(
			"nmon needs at least a width of 79 columns.",
		)
		return lipgloss.Place(a.width, a.height, lipgloss.Center, lipgloss.Center, msg)
	}

	if a.showHelp {
		return a.viewHelp()
	}
	if a.zoomActive {
		return a.viewZoom()
	}

	switch a.view {
	case viewExperiments:
		return a.viewExperiments()
	case viewLeaderboard:
		return a.viewLeaderboard()
	default:
		return a.viewDashboard()
	}
}

func (a *App) tickCmd() tea.Cmd {
	d := a.opts.RefreshEvery
	if d <= 0 {
		d = 2 * time.Second
	}
	return tea.Tick(d, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func copyCursors(in map[string]int64) map[string]int64 {
	out := make(map[string]int64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func (a *App) selectRun(run string) {
	a.currentRun = run
	a.series = make(map[string][]store.Point)
	a.cursors = make(map[string]int64)
	a.gpuLatest = nil
	a.routerLatest = nil
	a.summaryLatest = nil
	a.sqliteRun = nil
	a.gpuParsed = nil
	a.gpuTable = nil
	a.routerLayers = nil
	a.routerAgg = nil
	a.zoomActive = false
	a.markAllDashCardsDirty()
	a.markRightSidebarDirty()
	a.rebuildRightSidebarCache()
}

func (a *App) updateExperimentsKeys(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch k.String() {
	case "esc":
		a.view = viewDashboard
		return a, nil
	case "up":
		if a.runsCursor > 0 {
			a.runsCursor--
		}
		return a, a.maybeProbeVisibleRuns()
	case "down":
		if a.runsCursor < len(a.runs)-1 {
			a.runsCursor++
		}
		if len(a.runs) > 0 && a.runsCursor >= len(a.runs)-3 && !a.runsLoading && len(a.runs) >= a.runsLimit {
			a.runsLimit += 50
			return a, a.cmdLoadRuns(a.runsLimit)
		}
		return a, a.maybeProbeVisibleRuns()
	case "enter":
		if a.runsCursor >= 0 && a.runsCursor < len(a.runs) {
			run := a.runs[a.runsCursor].Run
			a.selectRun(run)
			a.view = viewDashboard
			return a, tea.Batch(
				a.cmdLoadSqliteRun(run),
				a.cmdPoll(run, copyCursors(a.cursors), pollFast),
			)
		}
		return a, nil
	case "r":
		if !a.runsLoading {
			a.runsLimit = 50
			a.runsCursor = 0
			a.runs = nil
			a.runStatsCache = make(map[string]store.RunSummary)
		}
		return a, tea.Batch(a.cmdLoadRuns(a.runsLimit), a.cmdLoadSqliteRuns(200))
	}
	return a, nil
}

func (a *App) experimentsVisibleRange() (start, end int) {
	w, h := a.width, a.height
	if w <= 0 || h <= 0 {
		return 0, 0
	}
	contentH := h - statusH
	if contentH < 0 {
		contentH = 0
	}
	linesBeforeRows := 5 // title, hint, blank, header, showing
	maxRows := contentH - linesBeforeRows - 2
	if maxRows < 1 {
		maxRows = 1
	}
	start = 0
	if a.runsCursor >= maxRows {
		start = a.runsCursor - maxRows + 1
	}
	end = min(len(a.runs), start+maxRows)
	return start, end
}

func (a *App) maybeProbeVisibleRuns() tea.Cmd {
	if a.runsProbeInFlight || len(a.runs) == 0 {
		return nil
	}
	start, end := a.experimentsVisibleRange()
	if start >= end {
		return nil
	}
	const probeCap = 15
	toProbe := make([]string, 0, probeCap)
	for i := start; i < end && len(toProbe) < probeCap; i++ {
		r := a.runs[i]
		if r.Run == "" {
			continue
		}
		if r.LastStep >= 0 {
			continue
		}
		if cached, ok := a.runStatsCache[r.Run]; ok && cached.LastStep >= 0 {
			a.runs[i].LastStep = cached.LastStep
			if cached.LastTsMs > 0 {
				a.runs[i].LastTsMs = cached.LastTsMs
			}
			continue
		}
		toProbe = append(toProbe, r.Run)
	}
	if len(toProbe) == 0 {
		return nil
	}
	return a.cmdProbeRuns(toProbe)
}

func (a *App) statusLine() string {
	var parts []string
	now := time.Now()
	view := "dash"
	if a.view == viewExperiments {
		view = "runs"
	}
	if a.view == viewLeaderboard {
		view = "leaderboard"
	}
	if a.zoomActive {
		view = "zoom"
	}
	parts = append(parts, fmt.Sprintf("view=%s", view))
	if a.currentRun != "" {
		parts = append(parts, fmt.Sprintf("run=%s", a.currentRun))
	}
	if !a.lastUpdate.IsZero() {
		parts = append(parts, "updated "+fmtAgo(now.Sub(a.lastUpdate))+" ago")
	}
	if a.pollInFlight {
		parts = append(parts, "polling…")
	}
	if a.newestRunLoading {
		parts = append(parts, "newest…")
	}
	if a.runsLoading {
		parts = append(parts, "runs…")
	}
	if a.sqliteRunsLoading {
		parts = append(parts, "experiments…")
	}
	if a.runsProbeInFlight {
		parts = append(parts, "probe…")
	}
	if a.err != nil {
		parts = append(parts, lipgloss.NewStyle().Foreground(lipgloss.Color("203")).Render("err: "+a.err.Error()))
	}
	parts = append(parts, "d:dash e:runs l:leaderboard r:refresh ?:help q:quit")
	if a.view == viewDashboard && !a.zoomActive {
		parts = append(parts, "z:zoom")
	}
	if a.opts.Debug {
		lay := computeLayout(a.width, a.height)
		parts = append(parts, fmt.Sprintf("[%dx%d L:%d M:%d R:%d]", a.width, a.height, lay.leftW, lay.mainW, lay.rightW))
		parts = append(parts, "t_poll="+fmtDur(a.lastPollDur))
		parts = append(parts, "t_probe="+fmtDur(a.lastProbeRunsDur))
		parts = append(parts, "t_runs="+fmtDur(a.lastListRunsDur))
	}
	s := lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Render(strings.Join(parts, "  "))
	return lipgloss.NewStyle().Width(a.width).Height(statusH).Padding(0, 1).Background(lipgloss.Color("236")).Render(s)
}

func fmtAgo(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	if d < 10*time.Second {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	if d < time.Minute {
		return fmt.Sprintf("%ds", int64(d.Round(time.Second).Seconds()))
	}
	if d < 10*time.Minute {
		m := int64(d / time.Minute)
		s := int64((d % time.Minute).Round(time.Second) / time.Second)
		return fmt.Sprintf("%dm%02ds", m, s)
	}
	return fmt.Sprintf("%dm", int64(d/time.Minute))
}

func fmtDur(d time.Duration) string {
	if d <= 0 {
		return "—"
	}
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.2fs", d.Seconds())
}

func (a *App) viewHelp() string {
	w, h := a.width, a.height
	if w <= 0 || h <= 0 {
		return ""
	}
	contentH := h - statusH
	if contentH < 0 {
		contentH = 0
	}

	title := panelTitle.Render("Help")
	lines := []string{title, ""}

	lines = append(lines,
		dim.Render("Global"),
		"  ?: help",
		"  q / ctrl+c: quit",
		"",
		dim.Render("Dashboard"),
		"  d: dashboard",
		"  r: refresh now",
		"  z: zoom charts",
		"",
		dim.Render("Runs"),
		"  e: runs list",
		"  ↑/↓: select",
		"  enter: monitor run",
		"  r: reload list",
		"  esc: back",
		"",
		dim.Render("Leaderboard"),
		"  l: speedrun leaderboard",
		"  r: reload",
		"  esc/d: back",
		"",
		dim.Render("Notes"),
		"  steps for runs are lazily probed for the visible page and cached",
	)

	body := panelBorder.Width(w).Height(contentH).Padding(0, 1).Render(strings.Join(lines, "\n"))
	return lipgloss.JoinVertical(lipgloss.Left, body, a.statusLine())
}

func (a *App) updateZoomKeys(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch k.String() {
	case "esc", "enter", "z":
		a.zoomActive = false
		return a, nil
	case "left", "h":
		if a.zoomCard > 0 {
			a.zoomCard--
		}
		return a, nil
	case "right", "l":
		if a.zoomCard < len(a.dashCards)-1 {
			a.zoomCard++
		}
		return a, nil
	case "+", "=":
		if a.zoomN > 50 {
			a.zoomN = int(float64(a.zoomN) * 0.7)
			if a.zoomN < 50 {
				a.zoomN = 50
			}
		}
		return a, nil
	case "-", "_":
		if a.zoomN < 20000 {
			a.zoomN = int(float64(a.zoomN) * 1.4)
			if a.zoomN > 20000 {
				a.zoomN = 20000
			}
		}
		return a, nil
	}
	return a, nil
}

func (a *App) markAllDashCardsDirty() {
	a.dashMainDirty = true
	a.dashMainW = 0
	a.dashMainH = 0
	a.dashMain = ""
	for i := range a.dashCards {
		a.dashCards[i].dirty = true
	}
}

func (a *App) markDashCardDirty(tag string) {
	for i := range a.dashCards {
		if a.dashCards[i].spec.Tag == tag {
			a.dashCards[i].dirty = true
			a.dashMainDirty = true
			return
		}
	}
}

func (a *App) markRightSidebarDirty() {
	a.rightDirty = true
	a.rightW = 0
	a.rightH = 0
	a.right = ""
}
