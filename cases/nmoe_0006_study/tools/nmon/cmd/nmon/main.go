package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime/debug"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/Noumena-Network/nmoe/tools/nmon/internal/store"
	"github.com/Noumena-Network/nmoe/tools/nmon/internal/tui"
)

func getenvOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	defer func() {
		if r := recover(); r != nil {
			_, _ = fmt.Fprintf(os.Stderr, "panic: %v\n%s\n", r, debug.Stack())
			os.Exit(2)
		}
	}()

	var (
		// Local mode
		metricsDir      = flag.String("metrics-dir", getenvOr("NMOE_METRICS_DIR", "/data/metrics"), "metrics root dir (contains {run}/step_*.parquet)")
		experimentsDB   = flag.String("experiments-db", getenvOr("NMOE_EXPERIMENTS_DB", "/data/experiments.db"), "experiments sqlite db path")
		leaderboardPath = flag.String("leaderboard", getenvOr("NMOE_LEADERBOARD", ""), "speedrun leaderboard JSON path")
		runID           = flag.String("run", os.Getenv("NMOE_RUN"), "run id (defaults to $NMOE_RUN or newest)")

		// K8s remote mode
		k8sEnabled    = flag.Bool("k8s", os.Getenv("NMOE_K8S") != "", "enable k8s remote mode (exec into pod)")
		k8sNamespace  = flag.String("k8s-namespace", getenvOr("NMOE_K8S_NAMESPACE", "default"), "k8s namespace")
		k8sPod        = flag.String("k8s-pod", os.Getenv("NMOE_K8S_POD"), "k8s pod name to exec into")
		k8sSelector   = flag.String("k8s-selector", getenvOr("NMOE_K8S_SELECTOR", "app=nmoe,stage=debug"), "k8s label selector to find pod")
		k8sContainer  = flag.String("k8s-container", os.Getenv("NMOE_K8S_CONTAINER"), "k8s container name (optional)")
		k8sKubeconfig = flag.String("kubeconfig", os.Getenv("KUBECONFIG"), "path to kubeconfig (optional)")

		// Common
		refresh       = flag.Duration("refresh", 2*time.Second, "poll interval")
		initialPoints = flag.Int("initial-points", 2000, "initial points per chart to load")
		maxPoints     = flag.Int("max-points", 4000, "max points to keep per chart in memory")

		// Debug
		check = flag.Bool("check", false, "test k8s/store connection and exit (no TUI)")
	)
	flag.Parse()

	// --check: test connection without TUI
	if *check {
		if err := runCheck(*k8sEnabled, *k8sNamespace, *k8sPod, *k8sSelector, *k8sContainer, *k8sKubeconfig, *metricsDir); err != nil {
			fmt.Fprintf(os.Stderr, "check failed: %v\n", err)
			os.Exit(1)
		}
		os.Exit(0)
	}

	debugUI := os.Getenv("NMON_DEBUG") != ""
	m := tui.NewApp(tui.Options{
		MetricsDir:      *metricsDir,
		ExperimentsDB:   *experimentsDB,
		LeaderboardPath: *leaderboardPath,
		RunID:           *runID,
		K8sEnabled:     *k8sEnabled,
		K8sNamespace:   *k8sNamespace,
		K8sPodName:     *k8sPod,
		K8sPodSelector: *k8sSelector,
		K8sContainer:   *k8sContainer,
		K8sKubeconfig:  *k8sKubeconfig,
		RefreshEvery:   *refresh,
		InitialPoints:  *initialPoints,
		MaxPoints:      *maxPoints,
		Debug:          debugUI,
	})

	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseAllMotion())
	if _, err := p.Run(); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
}

func runCheck(k8sEnabled bool, namespace, pod, selector, container, kubeconfig, metricsDir string) error {
	fmt.Printf("k8s_enabled: %v\n", k8sEnabled)
	fmt.Printf("namespace: %s\n", namespace)
	fmt.Printf("pod: %s\n", pod)
	fmt.Printf("selector: %s\n", selector)
	fmt.Printf("container: %s\n", container)
	fmt.Printf("kubeconfig: %s\n", kubeconfig)
	fmt.Printf("metrics_dir: %s\n", metricsDir)

	if !k8sEnabled {
		fmt.Println("\nk8s mode disabled, checking local store...")
		s := store.NewDuckDBRank0(metricsDir)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		runs, err := s.ListRuns(ctx, 50)
		if err != nil {
			return fmt.Errorf("list runs: %w", err)
		}
		fmt.Printf("found %d runs\n", len(runs))
		for i, r := range runs {
			if i >= 5 {
				fmt.Printf("  ... and %d more\n", len(runs)-5)
				break
			}
			fmt.Printf("  - %s (step=%d)\n", r.Run, r.LastStep)
		}
		return nil
	}

	fmt.Println("\nk8s mode enabled, connecting...")
	s, err := store.NewK8sExecStore(store.K8sExecOptions{
		Namespace:   namespace,
		PodName:     pod,
		PodSelector: selector,
		Container:   container,
		MetricsDir:  metricsDir,
		Kubeconfig:  kubeconfig,
	})
	if err != nil {
		return fmt.Errorf("create k8s store: %w", err)
	}
	fmt.Printf("connected to pod: %s\n", s.PodName())

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fmt.Println("listing runs...")
	runs, err := s.ListRuns(ctx, 50)
	if err != nil {
		return fmt.Errorf("list runs: %w", err)
	}
	fmt.Printf("found %d runs\n", len(runs))
	for i, r := range runs {
		if i >= 5 {
			fmt.Printf("  ... and %d more\n", len(runs)-5)
			break
		}
		fmt.Printf("  - %s (step=%d)\n", r.Run, r.LastStep)
	}

	fmt.Println("\ntesting experiments store...")
	expStore := store.NewK8sExperimentsStoreFromExec(s, "/data/experiments.db")
	expRuns, err := expStore.ListRuns(50)
	if err != nil {
		return fmt.Errorf("list experiment runs: %w", err)
	}
	fmt.Printf("found %d experiment runs\n", len(expRuns))

	fmt.Println("\ncheck passed")
	return nil
}
