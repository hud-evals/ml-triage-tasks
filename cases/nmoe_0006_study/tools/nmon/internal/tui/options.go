package tui

import "time"

type Options struct {
	// Local mode
	MetricsDir      string
	ExperimentsDB   string
	LeaderboardPath string
	RunID           string

	// K8s remote mode
	K8sEnabled    bool
	K8sNamespace  string
	K8sPodName    string
	K8sPodSelector string
	K8sContainer  string
	K8sKubeconfig string

	RefreshEvery  time.Duration
	InitialPoints int
	MaxPoints     int

	Debug bool
}
