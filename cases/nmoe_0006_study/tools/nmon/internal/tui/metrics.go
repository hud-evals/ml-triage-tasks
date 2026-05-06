package tui

import (
	"fmt"
	"math"
	"strings"
)

type metricFmt func(v float64) string

func isFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func fmtFloat(digits int) metricFmt {
	return func(v float64) string {
		if !isFinite(v) {
			return "—"
		}
		return fmt.Sprintf("%.*f", digits, v)
	}
}

func fmtInt() metricFmt {
	return func(v float64) string {
		if !isFinite(v) {
			return "—"
		}
		return fmt.Sprintf("%d", int64(math.Round(v)))
	}
}

func fmtPct() metricFmt {
	return func(v float64) string {
		if !isFinite(v) {
			return "—"
		}
		if v <= 1.0 {
			v *= 100
		}
		return fmt.Sprintf("%.0f%%", v)
	}
}

func fmtTokPerS() metricFmt {
	return func(v float64) string {
		if !isFinite(v) {
			return "—"
		}
		if v >= 1e6 {
			return fmt.Sprintf("%.2fM", v/1e6)
		}
		if v >= 1e3 {
			return fmt.Sprintf("%.1fk", v/1e3)
		}
		return fmt.Sprintf("%.0f", v)
	}
}

func shortTag(tag string) string {
	if tag == "" {
		return ""
	}
	if i := strings.LastIndex(tag, "/"); i >= 0 && i < len(tag)-1 {
		return tag[i+1:]
	}
	return tag
}

type chartSpec struct {
	Title string
	Tag   string
	Fmt   metricFmt
}

var defaultCharts = []chartSpec{
	{Title: "loss", Tag: "train/loss", Fmt: fmtFloat(3)},
	{Title: "tok/s", Tag: "throughput/tokens_per_s_gpu", Fmt: fmtTokPerS()},
	{Title: "ms/step", Tag: "throughput/ms_per_step", Fmt: fmtFloat(1)},
	{Title: "tflops", Tag: "efficiency/tflops", Fmt: fmtFloat(1)},
	{Title: "fp8 tflops", Tag: "efficiency/fp8_tflops", Fmt: fmtFloat(1)},
	{Title: "bf16 tflops", Tag: "efficiency/bf16_tflops", Fmt: fmtFloat(1)},
	{Title: "grad", Tag: "train/grad_norm", Fmt: fmtFloat(2)},
	{Title: "lr", Tag: "optimizer/lr_dense", Fmt: fmtFloat(6)},
	{Title: "router cv", Tag: "router_agg/mean_cv", Fmt: fmtFloat(2)},
	{Title: "entropy", Tag: "router_agg/mean_entropy", Fmt: fmtFloat(2)},
	{Title: "dead exp", Tag: "router_agg/dead_experts_count", Fmt: fmtInt()},
	{Title: "alloc GiB", Tag: "memory/max_alloc_gib", Fmt: fmtFloat(1)},
}
