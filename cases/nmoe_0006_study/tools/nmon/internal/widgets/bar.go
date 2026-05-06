package widgets

import (
	"fmt"
	"math"
	"strings"
)

var fracBlocks = []rune{' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'}

func Bar(prefix string, percent float64, width int) string {
	if width <= 0 {
		return ""
	}
	label := prefix + ": "
	if len(label) >= width {
		return label[:width]
	}

	if math.IsNaN(percent) || math.IsInf(percent, 0) {
		text := "N/A"
		barW := width - len(label) - len(text) - 1
		if barW < 0 {
			barW = 0
		}
		return label + strings.Repeat("░", barW) + " " + text
	}

	if percent < 0 {
		percent = 0
	}
	if percent > 100 {
		percent = 100
	}

	text := fmt.Sprintf("%.0f%%", percent)
	if percent >= 100 {
		text = "MAX"
	}

	barW := width - len(label) - len(text) - 1
	if barW < 0 {
		barW = 0
	}

	units := int(math.Round(float64(barW) * 8.0 * (percent / 100.0)))
	if percent > 0 && units == 0 && barW > 0 {
		units = 1
	}
	full := units / 8
	rem := units % 8
	if full > barW {
		full = barW
		rem = 0
	}

	var b strings.Builder
	b.WriteString(label)
	b.WriteString(strings.Repeat("█", full))
	used := full
	if rem > 0 && used < barW {
		b.WriteRune(fracBlocks[rem])
		used++
	}
	if used < barW {
		b.WriteString(strings.Repeat(" ", barW-used))
	}
	b.WriteString(" ")
	b.WriteString(text)
	return b.String()
}
