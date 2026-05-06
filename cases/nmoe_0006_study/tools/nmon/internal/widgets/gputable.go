package widgets

import (
	"fmt"
	"strings"
)

// GpuInfo holds GPU state for rendering.
type GpuInfo struct {
	Index int
	Name  string
	Temp  float64
	Fan   float64 // percentage 0-100, -1 for N/A
	Power float64
	PwrL  float64
	MemU  float64 // GiB used
	MemT  float64 // GiB total
	Util  float64 // 0-100
}

// GpuTable renders an nvitop-style GPU table with optional side bars.
// Returns empty string if width < 79 or no GPUs.
func GpuTable(gpus []GpuInfo, width int) string {
	if len(gpus) == 0 || width < 79 {
		return ""
	}

	// Calculate if we have room for side bars (nvitop shows them at width >= 100)
	showBars := width >= 100
	remaining := width - 79
	var barW int
	if showBars {
		barW = (remaining - 3) / 2 // two bars with separator
		if barW < 8 {
			barW = 8
		}
		if barW > 30 {
			barW = 30
		}
	}

	var b strings.Builder

	// Top border
	b.WriteString("╒═══════════════════════════════════════════════════════════════════════════╕")
	if showBars {
		b.WriteString(strings.Repeat("═", remaining-1) + "╕")
	}
	b.WriteString("\n")

	// Title line
	title := "│  nmon GPU Monitor                                                         │"
	if showBars {
		title = title[:len(title)-1] + " " + strings.Repeat(" ", remaining-2) + "│"
	}
	b.WriteString(title + "\n")

	// Column headers separator
	b.WriteString("├───────────────────────────────┬──────────────────────┬────────────────────┤")
	if showBars {
		b.WriteString(strings.Repeat("─", barW) + "┬" + strings.Repeat("─", remaining-barW-2) + "┤")
	}
	b.WriteString("\n")

	// Column headers
	header := "│ GPU Fan Temp Perf Pwr:Usg/Cap │      Memory-Usage    │ GPU-Util  Mode     │"
	if showBars {
		memHdr := centerStr("MEM", barW)
		utlHdr := centerStr("UTL", remaining-barW-2)
		header = header[:len(header)-1] + memHdr + "│" + utlHdr + "│"
	}
	b.WriteString(header + "\n")

	// Header/data separator
	b.WriteString("╞═══════════════════════════════╪══════════════════════╪════════════════════╡")
	if showBars {
		b.WriteString(strings.Repeat("═", barW) + "╪" + strings.Repeat("═", remaining-barW-2) + "╡")
	}
	b.WriteString("\n")

	// GPU rows
	for i, g := range gpus {
		// Fan string
		fanStr := "N/A"
		if g.Fan >= 0 {
			if g.Fan >= 100 {
				fanStr = "MAX"
			} else {
				fanStr = fmt.Sprintf("%2.0f%%", g.Fan)
			}
		}

		// Temp string
		tempStr := fmt.Sprintf("%2.0fC", g.Temp)

		// Power string
		pwrStr := fmt.Sprintf("%3.0fW/%3.0fW", g.Power, g.PwrL)

		// Memory string in MiB
		memUsedMiB := g.MemU * 1024
		memTotalMiB := g.MemT * 1024
		memStr := fmt.Sprintf("%5.0fMiB/%5.0fMiB", memUsedMiB, memTotalMiB)

		// Memory percentage for bar
		memPct := 0.0
		if g.MemT > 0 {
			memPct = (g.MemU / g.MemT) * 100
		}

		// Util string
		utilStr := fmt.Sprintf("%3.0f%%", g.Util)

		// Main data line
		line := fmt.Sprintf("│ %3d %3s %4s  P0  %11s │ %18s │ %7s  Default │",
			g.Index, fanStr, tempStr, pwrStr, memStr, utilStr)

		if showBars {
			memBar := makeBarChart("MEM", memPct, barW-1)
			utlBar := makeBarChart("UTL", g.Util, remaining-barW-3)
			line = line[:len(line)-1] + memBar + "│" + utlBar + "│"
		}
		b.WriteString(line + "\n")

		// Row separator (not after last)
		if i < len(gpus)-1 {
			sep := "├───────────────────────────────┼──────────────────────┼────────────────────┤"
			if showBars {
				sep = sep[:len(sep)-1] + "┼" + strings.Repeat("─", barW) + "┼" + strings.Repeat("─", remaining-barW-2) + "┤"
			}
			b.WriteString(sep + "\n")
		}
	}

	// Bottom border
	b.WriteString("╘═══════════════════════════════╧══════════════════════╧════════════════════╛")
	if showBars {
		b.WriteString(strings.Repeat("═", barW) + "╧" + strings.Repeat("═", remaining-barW-2) + "╛")
	}

	return b.String()
}

// makeBarChart creates a bar like "MEM ████████░░░░░░░  75%"
func makeBarChart(label string, pct float64, width int) string {
	if width < 10 {
		return strings.Repeat(" ", width)
	}
	if pct < 0 {
		pct = 0
	}
	if pct > 100 {
		pct = 100
	}

	// Format: "LBL ████░░░░ XXX%"
	labelW := len(label) + 1 // "MEM "
	pctW := 5                // " XXX%"
	barW := width - labelW - pctW
	if barW < 1 {
		barW = 1
	}

	filled := int((pct / 100.0) * float64(barW))
	if pct > 0 && filled == 0 {
		filled = 1
	}
	if filled > barW {
		filled = barW
	}

	pctStr := fmt.Sprintf("%3.0f%%", pct)
	if pct >= 100 {
		pctStr = " MAX"
	}

	return fmt.Sprintf("%s %s%s%s",
		label,
		strings.Repeat("█", filled),
		strings.Repeat("░", barW-filled),
		pctStr)
}

func centerStr(s string, width int) string {
	if len(s) >= width {
		return s[:width]
	}
	pad := width - len(s)
	left := pad / 2
	right := pad - left
	return strings.Repeat(" ", left) + s + strings.Repeat(" ", right)
}
