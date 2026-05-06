package widgets

import (
	"math"
	"strings"
)

var sparkRunes = []rune("▁▂▃▄▅▆▇█")

func Spark(values []float64, width int) string {
	if width <= 0 {
		return ""
	}
	if len(values) == 0 {
		return strings.Repeat(" ", width)
	}

	minV := math.Inf(1)
	maxV := math.Inf(-1)
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			continue
		}
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	if math.IsNaN(minV) || math.IsInf(minV, 0) || math.IsNaN(maxV) || math.IsInf(maxV, 0) {
		return strings.Repeat(" ", width)
	}
	if maxV == minV {
		return strings.Repeat(string(sparkRunes[len(sparkRunes)-1]), min(width, len(values))) + strings.Repeat(" ", max(0, width-min(width, len(values))))
	}

	out := make([]rune, 0, width)
	for i := 0; i < width; i++ {
		idx := (i * len(values)) / width
		v := values[idx]
		if math.IsNaN(v) || math.IsInf(v, 0) {
			out = append(out, ' ')
			continue
		}
		n := (v - minV) / (maxV - minV)
		level := int(math.Round(n * float64(len(sparkRunes)-1)))
		if level < 0 {
			level = 0
		}
		if level >= len(sparkRunes) {
			level = len(sparkRunes) - 1
		}
		out = append(out, sparkRunes[level])
	}
	return string(out)
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
