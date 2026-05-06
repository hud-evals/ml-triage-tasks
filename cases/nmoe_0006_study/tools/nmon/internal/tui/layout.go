package tui

// Layout constants matching leet's phi-based proportions.
const (
	statusH = 1

	// Golden ratio sidebar widths (same as leet).
	// When both sidebars visible: (1 - 1/phi) / phi ≈ 0.236
	sidebarWidthRatio = 0.236
	sidebarMinW       = 40
	sidebarMaxW       = 120
)

type layout struct {
	leftW  int
	mainW  int
	rightW int
	h      int
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func computeLayout(w, h int) layout {
	contentH := h - statusH
	if contentH < 0 {
		contentH = 0
	}

	// Calculate sidebar width based on terminal width
	sideW := int(float64(w) * sidebarWidthRatio)
	sideW = clamp(sideW, sidebarMinW, sidebarMaxW)

	// Safety: if sidebars would leave no room for main, shrink them
	const minMainW = 10
	totalSideW := sideW * 2
	if totalSideW >= w-minMainW {
		// Terminal too narrow - disable sidebars
		if totalSideW >= w {
			sideW = 0
		} else {
			// Shrink proportionally
			sideW = (w - minMainW) / 2
		}
	}

	mainW := w - sideW*2
	if mainW < 0 {
		mainW = 0
	}

	return layout{
		leftW:  sideW,
		mainW:  mainW,
		rightW: sideW,
		h:      contentH,
	}
}
