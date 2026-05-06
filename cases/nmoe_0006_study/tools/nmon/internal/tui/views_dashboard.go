package tui

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/Noumena-Network/nmoe/tools/nmon/internal/store"
	"github.com/Noumena-Network/nmoe/tools/nmon/internal/widgets"
)

func (a *App) viewDashboard() string {
	lay := computeLayout(a.width, a.height)

	left := a.renderLeftSidebar(lay.leftW, lay.h)
	main := a.renderChartsGrid(lay.mainW, lay.h)
	right := a.renderRightSidebar(lay.rightW, lay.h)

	content := lipgloss.JoinHorizontal(lipgloss.Top, left, main, right)
	statusBar := a.statusLine()
	fullView := lipgloss.JoinVertical(lipgloss.Left, content, statusBar)

	// Use Place to enforce exact terminal dimensions (like leet)
	return lipgloss.Place(a.width, a.height, lipgloss.Left, lipgloss.Top, fullView)
}

func (a *App) viewZoom() string {
	w, h := a.width, a.height
	if w <= 0 || h <= 0 {
		return ""
	}
	contentH := h - statusH
	if contentH < 0 {
		contentH = 0
	}

	if len(a.dashCards) == 0 {
		msg := panelBorder.Width(w).Height(contentH).Padding(0, 1).Render(panelTitle.Render("Zoom") + "\n" + dim.Render("no charts"))
		return lipgloss.JoinVertical(lipgloss.Left, msg, a.statusLine())
	}
	if a.zoomCard < 0 {
		a.zoomCard = 0
	}
	if a.zoomCard >= len(a.dashCards) {
		a.zoomCard = len(a.dashCards) - 1
	}
	spec := a.dashCards[a.zoomCard].spec
	tag := spec.Tag
	pts := a.series[tag]

	title := panelTitle.Render("Zoom: " + spec.Title)
	hints := dim.Render("←/→ switch  +/- zoom  esc exit")

	var header string
	if len(pts) > 0 {
		last := pts[len(pts)-1]
		val := spec.Fmt(last.Value)
		header = fmt.Sprintf("tag=%s  step=%d  val=%s  window=%d", tag, last.Step, val, a.zoomN)
	} else {
		header = fmt.Sprintf("tag=%s  window=%d", tag, a.zoomN)
	}

	lines := []string{title, hints, dim.Render(header), ""}

	cardH := contentH - len(lines) - 1
	if cardH < 3 {
		cardH = 3
	}
	cardW := w

	var window []store.Point
	if len(pts) > 0 {
		n := a.zoomN
		if n <= 0 {
			n = 2000
		}
		if n > len(pts) {
			n = len(pts)
		}
		window = pts[len(pts)-n:]
	}

	card := panelBorder.Copy().Width(cardW).MaxWidth(cardW).Height(cardH).MaxHeight(cardH).Padding(0, 1)
	innerW := max(0, cardW-4)
	innerH := max(0, cardH-2)

	if len(window) == 0 {
		lines = append(lines, card.Render(dim.Render("no data")))
	} else {
		chartH := max(1, innerH-1)
		body := []string{panelTitle.Render(spec.Title)}
		if innerW >= 20 && chartH >= 6 {
			maxPts := max(2, innerW*4)
			xs, ys := downsampleXYMean(window, maxPts)
			chart := widgets.NewBrailleChart("", innerW, chartH)
			chart.SetData(xs, ys)
			body = append(body, chart.View())
		} else {
			values := downsampleMean(window, max(1, innerW))
			body = append(body, widgets.Spark(values, max(1, innerW)))
		}
		lines = append(lines, card.Render(strings.Join(body, "\n")))
	}

	body := lipgloss.JoinVertical(lipgloss.Left, lines...)
	return lipgloss.Place(w, h, lipgloss.Left, lipgloss.Top, lipgloss.JoinVertical(lipgloss.Left, body, a.statusLine()))
}

var (
	panelBorder = lipgloss.NewStyle().Border(lipgloss.NormalBorder()).BorderForeground(lipgloss.Color("240"))
	panelTitle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("252"))
	kvKey       = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	kvVal       = lipgloss.NewStyle().Foreground(lipgloss.Color("252"))
	dim         = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
)

func (a *App) renderLeftSidebar(w, h int) string {
	title := panelTitle.Render("Run")
	run := a.currentRun
	if run == "" {
		run = "loading..."
	}
	status := "—"
	exp := "—"
	started := "—"
	ended := "—"
	git := "—"
	dirty := "—"
	if a.sqliteRun != nil {
		status = a.sqliteRun.Status
		exp = a.sqliteRun.ExperimentID
		started = a.sqliteRun.StartedAt
		if a.sqliteRun.EndedAt != nil {
			ended = *a.sqliteRun.EndedAt
		}
		if a.sqliteRun.GitHash != nil {
			git = *a.sqliteRun.GitHash
		}
		if a.sqliteRun.GitDirty != nil {
			if *a.sqliteRun.GitDirty != 0 {
				dirty = "yes"
			} else {
				dirty = "no"
			}
		}
	} else if r, ok := a.sqliteRuns[a.currentRun]; ok {
		status = r.Status
		exp = r.ExperimentID
		started = r.StartedAt
		if r.EndedAt != nil {
			ended = *r.EndedAt
		}
		if r.GitHash != nil {
			git = *r.GitHash
		}
		if r.GitDirty != nil {
			if *r.GitDirty != 0 {
				dirty = "yes"
			} else {
				dirty = "no"
			}
		}
	}

	lines := []string{
		title,
		kv("id", run),
		kv("status", status),
		kv("experiment", exp),
		kv("started", started),
		kv("ended", ended),
		kv("git", git),
		kv("dirty", dirty),
		"",
		panelTitle.Render("Summary"),
	}

	type srow struct {
		tag   string
		label string
		fmt   metricFmt
	}
	srows := []srow{
		{tag: "train/loss", label: "loss", fmt: fmtFloat(3)},
		{tag: "throughput/tokens_per_s_gpu", label: "tok/s", fmt: fmtTokPerS()},
		{tag: "throughput/ms_per_step", label: "ms/step", fmt: fmtFloat(1)},
		{tag: "efficiency/tflops", label: "tflops", fmt: fmtFloat(1)},
		{tag: "gpu_agg/mean_utilization_gpu", label: "gpu util", fmt: fmtPct()},
		{tag: "gpu_agg/total_memory_used_gib", label: "gpu mem", fmt: fmtFloat(1)},
		{tag: "gpu_agg/max_temperature_c", label: "gpu temp", fmt: fmtFloat(0)},
	}
	for _, r := range srows {
		p, ok := a.summaryLatest[r.tag]
		if !ok {
			lines = append(lines, kv(r.label, "—"))
			continue
		}
		lines = append(lines, kv(r.label, r.fmt(p.Value)))
	}

	content := strings.Join(lines, "\n")
	return panelBorder.
		Width(w).MaxWidth(w).
		Height(h).MaxHeight(h).
		Padding(0, 1).
		Render(content)
}

func kv(k, v string) string {
	return lipgloss.JoinHorizontal(lipgloss.Left, kvKey.Render(k+":"), " ", kvVal.Render(v))
}

func (a *App) renderChartsGrid(w, h int) string {
	rows, cols := dashRows, dashCols
	if w <= 0 || h <= 0 {
		return ""
	}

	if len(a.dashCards) != rows*cols {
		a.dashCards = newDashCards()
		a.dashMainDirty = true
	}

	if a.dashMainW != w || a.dashMainH != h {
		a.dashMainW = w
		a.dashMainH = h
		a.dashMainDirty = true
		for i := range a.dashCards {
			a.dashCards[i].dirty = true
		}
	}
	if !a.dashMainDirty && a.dashMain != "" {
		return a.dashMain
	}

	colWs := splitSizes(w, cols)
	rowHs := splitSizes(h, rows)

	idx := 0
	var renderedRows []string
	for r := 0; r < rows; r++ {
		var cells []string
		for c := 0; c < cols; c++ {
			var cell string
			if idx < len(a.dashCards) {
				cell = a.dashCards[idx].render(a.series, colWs[c], rowHs[r])
			}
			cells = append(cells, cell)
			idx++
		}
		renderedRows = append(renderedRows, lipgloss.JoinHorizontal(lipgloss.Top, cells...))
	}
	a.dashMain = lipgloss.JoinVertical(lipgloss.Left, renderedRows...)
	a.dashMainDirty = false
	return a.dashMain
}

const (
	dashRows = 4
	dashCols = 3
)

type dashCardCache struct {
	spec chartSpec

	w int
	h int

	lastLen  int
	lastStep int64

	dirty bool

	rendered string

	chart  *widgets.BrailleChart
	chartW int
	chartH int
}

func newDashCards() []dashCardCache {
	n := dashRows * dashCols
	out := make([]dashCardCache, n)
	for i := 0; i < n; i++ {
		var spec chartSpec
		if i < len(defaultCharts) {
			spec = defaultCharts[i]
		} else {
			spec = chartSpec{Title: "", Tag: "", Fmt: fmtFloat(2)}
		}
		out[i] = dashCardCache{
			spec:  spec,
			dirty: true,
		}
	}
	return out
}

func (c *dashCardCache) render(series map[string][]store.Point, w, h int) string {
	if w <= 0 || h <= 0 {
		return ""
	}

	var pts []store.Point
	if c.spec.Tag != "" {
		pts = series[c.spec.Tag]
	}

	if w != c.w || h != c.h {
		c.dirty = true
	}
	if len(pts) != c.lastLen {
		c.dirty = true
	} else if len(pts) > 0 && pts[len(pts)-1].Step != c.lastStep {
		c.dirty = true
	}

	if !c.dirty && c.rendered != "" {
		return c.rendered
	}

	c.w, c.h = w, h
	c.lastLen = len(pts)
	if len(pts) > 0 {
		c.lastStep = pts[len(pts)-1].Step
	} else {
		c.lastStep = 0
	}

	card := panelBorder.Copy().
		Width(w).MaxWidth(w).
		Height(h).MaxHeight(h).
		Padding(0, 1)
	if c.spec.Tag == "" {
		c.rendered = card.Render("")
		c.dirty = false
		return c.rendered
	}

	var (
		lastVal  float64
		lastStep int64
		has      bool
	)
	if len(pts) > 0 {
		lastVal = pts[len(pts)-1].Value
		lastStep = pts[len(pts)-1].Step
		has = true
	}

	innerW := max(0, w-4) // border (2) + padding (2)
	innerH := max(0, h-2) // border only

	title := panelTitle.Render(c.spec.Title)
	val := "—"
	if has {
		val = c.spec.Fmt(lastVal)
	}
	header := lipgloss.JoinHorizontal(lipgloss.Left, title, dim.Render(fmt.Sprintf("  step %d  ", lastStep)), kvVal.Render(val))

	lines := []string{header}

	if !has {
		lines = append(lines, dim.Render("no data"))
	} else {
		chartH := max(1, innerH-1)
		if innerW >= 20 && chartH >= 6 {
			maxPts := max(2, innerW*4)
			xs, ys := downsampleXYMean(pts, maxPts)
			if c.chart == nil {
				c.chart = widgets.NewBrailleChart("", innerW, chartH)
				c.chartW = innerW
				c.chartH = chartH
			} else if c.chartW != innerW || c.chartH != chartH {
				c.chart.Resize(innerW, chartH)
				c.chartW = innerW
				c.chartH = chartH
			}
			c.chart.SetData(xs, ys)
			lines = append(lines, c.chart.View())
		} else {
			sparkW := max(1, innerW)
			values := downsampleMean(pts, sparkW)
			lines = append(lines, widgets.Spark(values, sparkW))
		}
	}

	c.rendered = card.Render(strings.Join(lines, "\n"))
	c.dirty = false
	return c.rendered
}

func downsampleMean(pts []store.Point, n int) []float64 {
	if n <= 0 {
		return nil
	}
	if len(pts) == 0 {
		return []float64{}
	}
	if len(pts) <= n {
		out := make([]float64, 0, len(pts))
		for _, p := range pts {
			out = append(out, p.Value)
		}
		return out
	}

	out := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		start := (i * len(pts)) / n
		end := ((i + 1) * len(pts)) / n
		if end <= start {
			end = start + 1
		}
		if end > len(pts) {
			end = len(pts)
		}
		sum := 0.0
		cnt := 0.0
		for j := start; j < end; j++ {
			v := pts[j].Value
			if math.IsNaN(v) || math.IsInf(v, 0) {
				continue
			}
			sum += v
			cnt++
		}
		if cnt == 0 {
			out = append(out, 0)
		} else {
			out = append(out, sum/cnt)
		}
	}
	return out
}

func downsampleXYMean(pts []store.Point, n int) ([]float64, []float64) {
	if n <= 0 || len(pts) == 0 {
		return nil, nil
	}
	if len(pts) <= n {
		xs := make([]float64, 0, len(pts))
		ys := make([]float64, 0, len(pts))
		for _, p := range pts {
			if math.IsNaN(p.Value) || math.IsInf(p.Value, 0) {
				continue
			}
			xs = append(xs, float64(p.Step))
			ys = append(ys, p.Value)
		}
		return xs, ys
	}

	xs := make([]float64, 0, n)
	ys := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		start := (i * len(pts)) / n
		end := ((i + 1) * len(pts)) / n
		if end <= start {
			end = start + 1
		}
		if end > len(pts) {
			end = len(pts)
		}

		sum := 0.0
		cnt := 0.0
		for j := start; j < end; j++ {
			v := pts[j].Value
			if math.IsNaN(v) || math.IsInf(v, 0) {
				continue
			}
			sum += v
			cnt++
		}
		if cnt == 0 {
			continue
		}

		xs = append(xs, float64(pts[end-1].Step))
		ys = append(ys, sum/cnt)
	}
	return xs, ys
}

func (a *App) renderRightSidebar(w, h int) string {
	if w <= 0 || h <= 0 {
		return ""
	}

	if !a.rightDirty && a.rightW == w && a.rightH == h && a.right != "" {
		return a.right
	}
	a.right = a.buildRightSidebar(w, h)
	a.rightW = w
	a.rightH = h
	a.rightDirty = false
	return a.right
}

func (a *App) rebuildRightSidebarCache() {
	if a.width <= 0 || a.height <= 0 {
		return
	}
	lay := computeLayout(a.width, a.height)
	if lay.rightW <= 0 || lay.h <= 0 {
		return
	}
	a.right = a.buildRightSidebar(lay.rightW, lay.h)
	a.rightW = lay.rightW
	a.rightH = lay.h
	a.rightDirty = false
}

func (a *App) buildRightSidebar(w, h int) string {
	if w <= 0 || h <= 0 {
		return ""
	}
	gpuH := max(10, h/2)
	if gpuH > h {
		gpuH = h
	}
	routerH := h - gpuH
	if routerH < 0 {
		routerH = 0
	}

	// Use MaxWidth/MaxHeight to enforce truncation (like leet does)
	gpu := panelBorder.
		Width(w).MaxWidth(w).
		Height(gpuH).MaxHeight(gpuH).
		Padding(0, 1).
		Render(a.renderGpuPanel(w-4, gpuH-2))
	router := panelBorder.
		Width(w).MaxWidth(w).
		Height(routerH).MaxHeight(routerH).
		Padding(0, 1).
		Render(a.renderRouterPanel(w-4, routerH-2))
	return lipgloss.JoinVertical(lipgloss.Left, gpu, router)
}

type gpuInfo struct {
	Index int
	Util  float64
	MemU  float64
	MemT  float64
	Pwr   float64
	PwrL  float64
	Temp  float64
	Fan   float64
}

func (a *App) renderGpuPanel(w, h int) string {
	if h <= 0 {
		return ""
	}
	title := panelTitle.Render("GPU")
	if len(a.gpuParsed) == 0 {
		return strings.Join([]string{title, dim.Render("no gpu metrics")}, "\n")
	}
	gpus := a.gpuParsed
	if len(gpus) == 0 {
		return strings.Join([]string{title, dim.Render("no gpu metrics")}, "\n")
	}

	// Use nvitop-style table if we have enough width (79 chars for full nvitop)
	if w >= 79 {
		table := widgets.GpuTable(a.gpuTable, w)
		if table != "" {
			return table // nvitop table has its own header
		}
	}

	// Fallback to simple format for narrow terminals
	lines := []string{title}
	header := fmt.Sprintf("%-3s %-5s %-9s %-15s %-5s", "id", "temp", "pwr", "mem", "util")
	lines = append(lines, dim.Render(header))
	maxRows := h - 2
	if maxRows < 1 {
		maxRows = 1
	}
	for i := 0; i < len(gpus) && i < maxRows; i++ {
		g := gpus[i]
		memPct := 0.0
		if g.MemT > 0 {
			memPct = (g.MemU / g.MemT) * 100
		}
		util := g.Util
		if util <= 1 {
			util *= 100
		}
		tempS := colorTemp(g.Temp).Render(fmt.Sprintf("%.0fC", g.Temp))
		pwrS := fmt.Sprintf("%.0f/%.0fW", g.Pwr, g.PwrL)
		memS := fmt.Sprintf("%.0f/%.0fGiB", g.MemU, g.MemT)
		utilS := colorUtil(util).Render(fmt.Sprintf("%.0f%%", util))
		row := fmt.Sprintf("%-3d %-5s %-9s %-15s %-5s", g.Index, tempS, pwrS, memS, utilS)
		if w >= 44 {
			barW := min(20, w-24)
			row += " " + colorUtil(memPct).Render(widgets.Bar("MEM", memPct, barW))
		}
		lines = append(lines, row)
	}
	return strings.Join(lines, "\n")
}

func parseGpus(latest map[string]store.Point) []gpuInfo {
	byIdx := map[int]*gpuInfo{}
	for tag, p := range latest {
		if !strings.HasPrefix(tag, "gpu/") {
			continue
		}
		parts := strings.Split(tag, "/")
		if len(parts) != 3 {
			continue
		}
		idx, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}
		key := parts[2]
		g := byIdx[idx]
		if g == nil {
			g = &gpuInfo{Index: idx}
			byIdx[idx] = g
		}
		switch key {
		case "utilization_gpu":
			g.Util = p.Value
		case "memory_used_gib":
			g.MemU = p.Value
		case "memory_total_gib":
			g.MemT = p.Value
		case "power_draw_w":
			g.Pwr = p.Value
		case "power_limit_w":
			g.PwrL = p.Value
		case "temperature_c":
			g.Temp = p.Value
		case "fan_speed":
			g.Fan = p.Value
		}
	}
	out := make([]gpuInfo, 0, len(byIdx))
	for _, g := range byIdx {
		out = append(out, *g)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Index < out[j].Index })
	return out
}

type routerLayer struct {
	Idx          int
	CV           float64
	Entropy      float64
	Experts      float64
	BiasRange    float64
	MaxLoad      float64
	HasAnyMetric bool
}

func (a *App) renderRouterPanel(w, h int) string {
	if h <= 0 {
		return ""
	}
	title := panelTitle.Render("Router")
	if len(a.routerLayers) == 0 && len(a.routerAgg) == 0 {
		return strings.Join([]string{title, dim.Render("no router metrics")}, "\n")
	}
	layers, agg := a.routerLayers, a.routerAgg
	lines := []string{title}

	// Comm/RDEP health (rank-0 only when emitted). Keep this compact.
	if len(a.summaryLatest) > 0 {
		comm := a.renderCommHealthLine()
		if comm != "" {
			lines = append(lines, dim.Render(comm))
		}
	}
	if len(agg) > 0 {
		lines = append(lines, dim.Render(strings.Join(agg, "  ")))
	}
	if len(layers) == 0 {
		lines = append(lines, dim.Render("no layer metrics"))
		return strings.Join(lines, "\n")
	}
	lines = append(lines, dim.Render(fmt.Sprintf("%-5s %-6s %-7s %-9s", "layer", "cv", "entropy", "experts")))
	maxRows := h - len(lines)
	if maxRows < 1 {
		maxRows = 1
	}
	for i := 0; i < len(layers) && i < maxRows; i++ {
		l := layers[i]
		cvS := colorCV(l.CV).Render(fmt.Sprintf("%.2f", l.CV))
		entS := colorEntropy(l.Entropy).Render(fmt.Sprintf("%.2f", l.Entropy))
		expS := colorExperts(l.Experts).Render(fmt.Sprintf("%.0f", l.Experts))
		lines = append(lines, fmt.Sprintf("%-5s %-6s %-7s %-9s", fmt.Sprintf("L%02d", l.Idx), cvS, entS, expS))
	}
	_ = w
	return strings.Join(lines, "\n")
}

func (a *App) renderCommHealthLine() string {
	get := func(tag string) (float64, bool) {
		p, ok := a.summaryLatest[tag]
		if !ok {
			return 0, false
		}
		return p.Value, true
	}

	parts := make([]string, 0, 6)
	if v, ok := get("comm/r0/capacity_utilization"); ok && isFinite(v) {
		parts = append(parts, "cap_util="+fmtPct()(v))
	}
	if v, ok := get("comm/r0/dropped_rows"); ok && isFinite(v) {
		parts = append(parts, fmt.Sprintf("drop=%d", int64(math.Round(v))))
	}
	if v, ok := get("comm/r0/M_recv"); ok && isFinite(v) {
		parts = append(parts, fmt.Sprintf("M_recv=%d", int64(math.Round(v))))
	}
	if v, ok := get("comm/r0/M_back"); ok && isFinite(v) {
		parts = append(parts, fmt.Sprintf("M_back=%d", int64(math.Round(v))))
	}
	if v, ok := get("throughput/loader_wait_ms"); ok && isFinite(v) {
		parts = append(parts, fmt.Sprintf("loader=%.0fms", v))
	}
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "  ")
}

func parseRouter(latest map[string]store.Point) ([]routerLayer, []string) {
	byIdx := map[int]*routerLayer{}
	var agg []string

	for tag, p := range latest {
		if strings.HasPrefix(tag, "router_agg/") {
			k := strings.TrimPrefix(tag, "router_agg/")
			switch k {
			case "mean_cv", "mean_entropy", "min_entropy", "dead_experts_count":
				agg = append(agg, fmt.Sprintf("%s=%v", k, round2(p.Value)))
			}
			continue
		}
		if !strings.HasPrefix(tag, "router/layer_") {
			continue
		}
		parts := strings.Split(tag, "/")
		if len(parts) != 3 {
			continue
		}
		layerPart := strings.TrimPrefix(parts[1], "layer_")
		idx, err := strconv.Atoi(layerPart)
		if err != nil {
			continue
		}
		key := parts[2]
		l := byIdx[idx]
		if l == nil {
			l = &routerLayer{Idx: idx}
			byIdx[idx] = l
		}
		switch key {
		case "cv":
			l.CV = p.Value
			l.HasAnyMetric = true
		case "entropy":
			l.Entropy = p.Value
			l.HasAnyMetric = true
		case "experts_active":
			l.Experts = p.Value
			l.HasAnyMetric = true
		case "bias_range":
			l.BiasRange = p.Value
			l.HasAnyMetric = true
		case "max_load":
			l.MaxLoad = p.Value
			l.HasAnyMetric = true
		}
	}

	layers := make([]routerLayer, 0, len(byIdx))
	for _, l := range byIdx {
		if !l.HasAnyMetric {
			continue
		}
		layers = append(layers, *l)
	}
	sort.Slice(layers, func(i, j int) bool { return layers[i].Idx < layers[j].Idx })
	sort.Strings(agg)
	return layers, agg
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }

func colorUtil(pct float64) lipgloss.Style {
	switch {
	case pct >= 80:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	case pct >= 50:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("215"))
	default:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("114"))
	}
}

func colorTemp(c float64) lipgloss.Style {
	switch {
	case c >= 80:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	case c >= 70:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("215"))
	default:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("114"))
	}
}

func colorCV(v float64) lipgloss.Style {
	switch {
	case v >= 2.0:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	case v >= 1.0:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("215"))
	default:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("114"))
	}
}

func colorEntropy(v float64) lipgloss.Style {
	switch {
	case v >= 0.8:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("114"))
	case v >= 0.6:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("215"))
	default:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	}
}

func colorExperts(v float64) lipgloss.Style {
	switch {
	case v >= 61:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("114"))
	case v >= 40:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("215"))
	default:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	}
}

func splitSizes(total, n int) []int {
	if n <= 0 {
		return nil
	}
	base := total / n
	rem := total % n
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = base
		if i < rem {
			out[i]++
		}
	}
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
