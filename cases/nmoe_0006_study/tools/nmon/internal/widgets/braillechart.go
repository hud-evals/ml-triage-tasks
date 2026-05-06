package widgets

import (
	"fmt"
	"math"
	"sort"

	"github.com/NimbleMarkets/ntcharts/canvas"
	"github.com/NimbleMarkets/ntcharts/canvas/graph"
	"github.com/NimbleMarkets/ntcharts/linechart"
	"github.com/charmbracelet/lipgloss"
)

const (
	defaultMaxX = 20
	defaultMaxY = 1
)

// BrailleChart renders time-series data using braille patterns for high resolution.
type BrailleChart struct {
	linechart.Model

	xData, yData []float64
	xMin, xMax   float64
	yMin, yMax   float64

	graphStyle lipgloss.Style
	title      string
	dirty      bool
}

// NewBrailleChart creates a new braille chart with the given dimensions.
func NewBrailleChart(title string, width, height int) *BrailleChart {
	if width < 1 {
		width = 1
	}
	if height < 1 {
		height = 1
	}

	c := &BrailleChart{
		Model: linechart.New(width, height, 0, defaultMaxX, 0, defaultMaxY,
			linechart.WithXYSteps(4, 4),
			linechart.WithYLabelFormatter(formatYLabel),
		),
		xData:      make([]float64, 0, 256),
		yData:      make([]float64, 0, 256),
		xMin:       math.Inf(1),
		xMax:       math.Inf(-1),
		yMin:       math.Inf(1),
		yMax:       math.Inf(-1),
		graphStyle: lipgloss.NewStyle().Foreground(lipgloss.Color("86")), // cyan
		title:      title,
		dirty:      true,
	}
	c.AxisStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	c.LabelStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	return c
}

// SetData replaces all data with the given points.
func (c *BrailleChart) SetData(xs, ys []float64) {
	if len(xs) != len(ys) {
		return
	}
	c.xData = append(c.xData[:0], xs...)
	c.yData = append(c.yData[:0], ys...)

	c.xMin, c.xMax = math.Inf(1), math.Inf(-1)
	c.yMin, c.yMax = math.Inf(1), math.Inf(-1)

	for i := range xs {
		if xs[i] < c.xMin {
			c.xMin = xs[i]
		}
		if xs[i] > c.xMax {
			c.xMax = xs[i]
		}
		if ys[i] < c.yMin {
			c.yMin = ys[i]
		}
		if ys[i] > c.yMax {
			c.yMax = ys[i]
		}
	}

	c.updateRanges()
	c.dirty = true
}

// AddPoint appends a single (x, y) point.
func (c *BrailleChart) AddPoint(x, y float64) {
	c.xData = append(c.xData, x)
	c.yData = append(c.yData, y)

	if x < c.xMin {
		c.xMin = x
	}
	if x > c.xMax {
		c.xMax = x
	}
	if y < c.yMin {
		c.yMin = y
	}
	if y > c.yMax {
		c.yMax = y
	}

	c.updateRanges()
	c.dirty = true
}

// Resize changes the chart dimensions.
func (c *BrailleChart) Resize(width, height int) {
	if width < 1 {
		width = 1
	}
	if height < 1 {
		height = 1
	}
	if c.Width() == width && c.Height() == height {
		return
	}
	c.Model.Resize(width, height)
	c.dirty = true
	c.updateRanges()
}

// SetStyle sets the line color/style.
func (c *BrailleChart) SetStyle(s lipgloss.Style) {
	c.graphStyle = s
	c.dirty = true
}

// Title returns the chart title.
func (c *BrailleChart) Title() string {
	return c.title
}

// updateRanges recalculates axis ranges based on data bounds.
func (c *BrailleChart) updateRanges() {
	if len(c.yData) == 0 {
		return
	}

	// Y range with padding
	valueRange := c.yMax - c.yMin
	padding := valueRange * 0.1
	if padding < 1e-6 {
		padding = 0.1
	}

	newYMin := c.yMin - padding
	newYMax := c.yMax + padding

	// Don't go negative for non-negative data
	if c.yMin >= 0 && newYMin < 0 {
		newYMin = 0
	}

	// X domain
	dataXMax := c.xMax
	if !isFinite(dataXMax) {
		dataXMax = 0
	}
	niceMax := dataXMax
	if niceMax < defaultMaxX {
		niceMax = defaultMaxX
	} else {
		niceMax = float64(((int(math.Ceil(niceMax)) + 9) / 10) * 10)
	}

	viewMin := c.xMin
	if !isFinite(viewMin) {
		viewMin = 0
	}

	c.SetYRange(newYMin, newYMax)
	c.SetViewYRange(newYMin, newYMax)
	c.SetXRange(0, niceMax)
	c.SetViewXRange(viewMin, niceMax)
	c.SetXYRange(c.MinX(), c.MaxX(), newYMin, newYMax)
}

// Draw renders the chart to its internal canvas.
func (c *BrailleChart) Draw() {
	c.Clear()
	c.DrawXYAxisAndLabel()

	if c.GraphWidth() <= 0 || c.GraphHeight() <= 0 {
		c.dirty = false
		return
	}
	if len(c.xData) == 0 || len(c.yData) == 0 {
		c.dirty = false
		return
	}

	// Find visible data range via binary search
	lb := sort.Search(len(c.xData), func(i int) bool { return c.xData[i] >= c.ViewMinX() })
	ub := sort.Search(len(c.xData), func(i int) bool { return c.xData[i] > c.ViewMaxX() })
	if ub-lb <= 0 {
		c.dirty = false
		return
	}

	// Build braille grid
	bGrid := graph.NewBrailleGrid(
		c.GraphWidth(),
		c.GraphHeight(),
		0, float64(c.GraphWidth()),
		0, float64(c.GraphHeight()),
	)

	// Scale factors
	xRange := c.ViewMaxX() - c.ViewMinX()
	yRange := c.ViewMaxY() - c.ViewMinY()
	if xRange <= 0 {
		xRange = 1
	}
	if yRange <= 0 {
		yRange = 1
	}
	xScale := float64(c.GraphWidth()) / xRange
	yScale := float64(c.GraphHeight()) / yRange

	// Convert visible data to canvas coordinates
	points := make([]canvas.Float64Point, 0, ub-lb)
	for i := lb; i < ub; i++ {
		x := (c.xData[i] - c.ViewMinX()) * xScale
		y := (c.yData[i] - c.ViewMinY()) * yScale

		if x >= 0 && x <= float64(c.GraphWidth()) && y >= 0 && y <= float64(c.GraphHeight()) {
			points = append(points, canvas.Float64Point{X: x, Y: y})
		}
	}

	// Draw lines between consecutive points
	if len(points) == 1 {
		gp := bGrid.GridPoint(points[0])
		bGrid.Set(gp)
	} else {
		for i := 0; i < len(points)-1; i++ {
			gp1 := bGrid.GridPoint(points[i])
			gp2 := bGrid.GridPoint(points[i+1])
			bresenhamLine(bGrid, gp1, gp2)
		}
	}

	// Render braille patterns to canvas
	startX := 0
	if c.YStep() > 0 {
		startX = c.Origin().X + 1
	}
	patterns := bGrid.BraillePatterns()
	graph.DrawBraillePatterns(&c.Canvas, canvas.Point{X: startX, Y: 0}, patterns, c.graphStyle)

	c.dirty = false
}

// View returns the rendered chart as a string.
func (c *BrailleChart) View() string {
	if c.dirty {
		c.Draw()
	}
	return c.Model.View()
}

// bresenhamLine draws a line using Bresenham's algorithm.
func bresenhamLine(bGrid *graph.BrailleGrid, p1, p2 canvas.Point) {
	dx := absInt(p2.X - p1.X)
	dy := absInt(p2.Y - p1.Y)

	sx := 1
	if p1.X > p2.X {
		sx = -1
	}
	sy := 1
	if p1.Y > p2.Y {
		sy = -1
	}

	err := dx - dy
	x, y := p1.X, p1.Y

	for {
		bGrid.Set(canvas.Point{X: x, Y: y})
		if x == p2.X && y == p2.Y {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x += sx
		}
		if e2 < dx {
			err += dx
			y += sy
		}
	}
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func isFinite(f float64) bool {
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}

// formatYLabel formats Y-axis labels with appropriate precision.
func formatYLabel(step int, v float64) string {
	absV := math.Abs(v)
	switch {
	case absV == 0:
		return "0"
	case absV >= 1e6:
		return fmt.Sprintf("%.1fM", v/1e6)
	case absV >= 1e3:
		return fmt.Sprintf("%.1fk", v/1e3)
	case absV >= 1:
		return fmt.Sprintf("%.1f", v)
	case absV >= 0.01:
		return fmt.Sprintf("%.2f", v)
	case absV >= 0.001:
		return fmt.Sprintf("%.3f", v)
	default:
		return fmt.Sprintf("%.4f", v)
	}
}
