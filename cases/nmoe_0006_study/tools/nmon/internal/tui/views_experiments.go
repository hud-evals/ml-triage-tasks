package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

func (a *App) viewExperiments() string {
	w, h := a.width, a.height
	if w <= 0 || h <= 0 {
		return ""
	}
	contentH := h - statusH
	if contentH < 0 {
		contentH = 0
	}

	title := panelTitle.Render("Runs (rank_0)")
	hint := dim.Render("↑/↓ select • enter monitor • esc back • r reload • scroll loads more")

	lines := []string{title, hint, ""}

	if len(a.runs) == 0 {
		if a.runsLoading {
			lines = append(lines, dim.Render("loading runs…"))
		} else {
			lines = append(lines, dim.Render("no runs found in metrics dir"))
		}
		body := panelBorder.Width(w).Height(contentH).Padding(0, 1).Render(strings.Join(lines, "\n"))
		return lipgloss.JoinVertical(lipgloss.Left, body, a.statusLine())
	}

	header := dim.Render(fmt.Sprintf("%-3s %-40s %-10s %-8s %-20s", "", "run", "status", "step", "started"))
	loadingSuffix := ""
	if a.runsLoading {
		loadingSuffix = " • loading…"
	}
	lines = append(lines, header, dim.Render(fmt.Sprintf("showing %d (limit %d)%s", len(a.runs), a.runsLimit, loadingSuffix)))

	maxRows := contentH - len(lines) - 2
	if maxRows < 1 {
		maxRows = 1
	}
	start := 0
	if a.runsCursor >= maxRows {
		start = a.runsCursor - maxRows + 1
	}
	end := min(len(a.runs), start+maxRows)

	for i := start; i < end; i++ {
		r := a.runs[i]
		status := "—"
		started := "—"
		if sr, ok := a.sqliteRuns[r.Run]; ok {
			status = sr.Status
			started = sr.StartedAt
		}
		prefix := "  "
		rowStyle := lipgloss.NewStyle()
		if i == a.runsCursor {
			prefix = "▸ "
			rowStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("252"))
		}
		step := "—"
		if r.LastStep >= 0 {
			step = fmt.Sprintf("%d", r.LastStep)
		}
		line := fmt.Sprintf("%s%-40s %-10s %-8s %-20s", prefix, r.Run, status, step, started)
		lines = append(lines, rowStyle.Render(line))
	}

	body := panelBorder.Width(w).Height(contentH).Padding(0, 1).Render(strings.Join(lines, "\n"))
	return lipgloss.JoinVertical(lipgloss.Left, body, a.statusLine())
}
