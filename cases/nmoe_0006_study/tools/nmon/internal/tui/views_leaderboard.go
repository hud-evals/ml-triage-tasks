package tui

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// LeaderboardEntry matches the JSON structure in LEADERBOARD.json
type LeaderboardEntry struct {
	Config       string  `json:"config"`
	Dtype        string  `json:"dtype"`
	FinalLoss    float64 `json:"final_loss"`
	CoreScore    float64 `json:"core_score"`
	Tokens       int64   `json:"tokens"`
	Steps        int64   `json:"steps"`
	WallTimeS    float64 `json:"wall_time_s"`
	TargetReached bool   `json:"target_reached"`
	Date         string  `json:"date"`
	ExperimentID string  `json:"experiment_id"`
}

type leaderboardFile struct {
	Runs []LeaderboardEntry `json:"runs"`
}

func (a *App) loadLeaderboard() {
	a.leaderboardLoaded = true
	a.leaderboard = nil

	path := a.opts.LeaderboardPath
	if path == "" {
		return
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var lf leaderboardFile
	if err := json.Unmarshal(data, &lf); err != nil {
		return
	}

	a.leaderboard = lf.Runs
}

func (a *App) updateLeaderboardKeys(k tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch k.String() {
	case "esc", "d":
		a.view = viewDashboard
		return a, nil
	case "r":
		a.leaderboardLoaded = false
		a.loadLeaderboard()
		return a, nil
	}
	return a, nil
}

func (a *App) viewLeaderboard() string {
	w, h := a.width, a.height
	if w <= 0 || h <= 0 {
		return ""
	}
	contentH := h - statusH
	if contentH < 0 {
		contentH = 0
	}

	title := panelTitle.Render("Speedrun Leaderboard")
	hint := dim.Render("r reload • esc/d back")

	lines := []string{title, hint, ""}

	if len(a.leaderboard) == 0 {
		if a.opts.LeaderboardPath == "" {
			lines = append(lines, dim.Render("leaderboard path not configured"))
		} else {
			lines = append(lines, dim.Render("no speedrun results yet"))
			lines = append(lines, dim.Render("run: n speedrun dense"))
		}
		body := panelBorder.Width(w).Height(contentH).Padding(0, 1).Render(strings.Join(lines, "\n"))
		return lipgloss.JoinVertical(lipgloss.Left, body, a.statusLine())
	}

	// Header
	header := dim.Render(fmt.Sprintf("  %-3s %-10s %-8s %8s %8s %10s %10s %-12s",
		"#", "Config", "Dtype", "Loss", "CORE", "Tokens", "Time", "Date"))
	lines = append(lines, header)
	lines = append(lines, dim.Render(strings.Repeat("─", 80)))

	maxRows := contentH - len(lines) - 2
	if maxRows < 1 {
		maxRows = 1
	}

	gold := lipgloss.NewStyle().Foreground(lipgloss.Color("220")).Bold(true)
	silver := lipgloss.NewStyle().Foreground(lipgloss.Color("252"))
	bronze := lipgloss.NewStyle().Foreground(lipgloss.Color("208"))

	for i, entry := range a.leaderboard {
		if i >= maxRows {
			break
		}

		tokens := "—"
		if entry.Tokens > 0 {
			tokens = fmt.Sprintf("%.1fB", float64(entry.Tokens)/1e9)
		}

		timeStr := "—"
		if entry.WallTimeS > 0 {
			timeStr = fmt.Sprintf("%.1fm", entry.WallTimeS/60)
		}

		lossStr := "—"
		if entry.FinalLoss > 0 {
			lossStr = fmt.Sprintf("%.4f", entry.FinalLoss)
		}

		coreStr := "—"
		if entry.CoreScore > 0 {
			coreStr = fmt.Sprintf("%.3f", entry.CoreScore)
		}

		date := "—"
		if len(entry.Date) >= 10 {
			date = entry.Date[:10]
		}

		line := fmt.Sprintf("  %-3d %-10s %-8s %8s %8s %10s %10s %-12s",
			i+1, entry.Config, entry.Dtype, lossStr, coreStr, tokens, timeStr, date)

		var style lipgloss.Style
		switch i {
		case 0:
			style = gold
		case 1:
			style = silver
		case 2:
			style = bronze
		default:
			style = lipgloss.NewStyle()
		}

		lines = append(lines, style.Render(line))
	}

	if len(a.leaderboard) > maxRows {
		lines = append(lines, dim.Render(fmt.Sprintf("  ... and %d more", len(a.leaderboard)-maxRows)))
	}

	body := panelBorder.Width(w).Height(contentH).Padding(0, 1).Render(strings.Join(lines, "\n"))
	return lipgloss.JoinVertical(lipgloss.Left, body, a.statusLine())
}
