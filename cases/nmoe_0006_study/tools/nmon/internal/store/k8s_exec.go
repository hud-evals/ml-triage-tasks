package store

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/remotecommand"
)

// Default Python path in nmoe pods (uv venv)
const defaultPythonPath = "/workspace/nmoe/.venv/bin/python3"

// K8sExecStore queries DuckDB via kubectl exec into a cluster pod.
type K8sExecStore struct {
	mu sync.Mutex

	namespace  string
	podName    string
	container  string
	metricsDir string
	pythonPath string

	clientset *kubernetes.Clientset
	config    *rest.Config
}

// K8sExecOptions configures the K8s exec store.
type K8sExecOptions struct {
	// Namespace to find pods in (default: "default")
	Namespace string
	// PodName to exec into. If empty, finds pod by label selector.
	PodName string
	// PodSelector is the label selector to find a pod (e.g. "app=nmoe,stage=debug")
	PodSelector string
	// Container name to exec into (optional, uses first container if empty)
	Container string
	// MetricsDir is the path to metrics inside the pod (default: "/data/metrics")
	MetricsDir string
	// Kubeconfig path (optional, uses default if empty)
	Kubeconfig string
}

// NewK8sExecStore creates a new store that queries DuckDB via k8s exec.
func NewK8sExecStore(opts K8sExecOptions) (*K8sExecStore, error) {
	// Load kubeconfig
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	if opts.Kubeconfig != "" {
		loadingRules.ExplicitPath = opts.Kubeconfig
	}

	configOverrides := &clientcmd.ConfigOverrides{}
	kubeConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, configOverrides)

	config, err := kubeConfig.ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("k8s: load config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("k8s: create clientset: %w", err)
	}

	namespace := opts.Namespace
	if namespace == "" {
		namespace = "default"
	}

	metricsDir := opts.MetricsDir
	if metricsDir == "" {
		metricsDir = "/data/metrics"
	}

	s := &K8sExecStore{
		namespace:  namespace,
		podName:    opts.PodName,
		container:  opts.Container,
		metricsDir: metricsDir,
		pythonPath: defaultPythonPath,
		clientset:  clientset,
		config:     config,
	}

	// If no pod name, find one by selector (fallback: any Running pod).
	if s.podName == "" {
		if opts.PodSelector != "" {
			pod, err := s.findPod(context.Background(), opts.PodSelector)
			if err == nil {
				s.podName = pod
			} else {
				pod, err2 := s.findPod(context.Background(), "")
				if err2 != nil {
					return nil, err
				}
				s.podName = pod
			}
		} else {
			pod, err := s.findPod(context.Background(), "")
			if err != nil {
				return nil, err
			}
			s.podName = pod
		}
	}

	return s, nil
}

// findPod finds a running pod matching the selector.
func (s *K8sExecStore) findPod(ctx context.Context, selector string) (string, error) {
	opts := metav1.ListOptions{
		FieldSelector: "status.phase=Running",
	}
	if selector != "" {
		opts.LabelSelector = selector
	}
	pods, err := s.clientset.CoreV1().Pods(s.namespace).List(ctx, opts)
	if err != nil {
		return "", fmt.Errorf("k8s: list pods: %w", err)
	}
	if len(pods.Items) == 0 {
		if selector != "" {
			return "", fmt.Errorf("k8s: no running pods matching selector %q", selector)
		}
		return "", fmt.Errorf("k8s: no running pods in namespace %q", s.namespace)
	}
	return pods.Items[0].Name, nil
}

// SetPod sets the pod to exec into.
func (s *K8sExecStore) SetPod(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.podName = name
}

// PodName returns the resolved pod name.
func (s *K8sExecStore) PodName() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.podName
}

func (s *K8sExecStore) exec(ctx context.Context, cmd []string, stdin []byte) (stdout, stderr []byte, _ error) {
	s.mu.Lock()
	podName := s.podName
	container := s.container
	s.mu.Unlock()

	if podName == "" {
		return nil, nil, fmt.Errorf("k8s: no pod configured")
	}

	req := s.clientset.CoreV1().RESTClient().Post().
		Resource("pods").
		Name(podName).
		Namespace(s.namespace).
		SubResource("exec")

	execOpts := &corev1.PodExecOptions{
		Command: cmd,
		Stdin:   stdin != nil,
		Stdout:  true,
		Stderr:  true,
	}
	if container != "" {
		execOpts.Container = container
	}

	req.VersionedParams(execOpts, scheme.ParameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(s.config, "POST", req.URL())
	if err != nil {
		return nil, nil, fmt.Errorf("k8s: create executor: %w", err)
	}

	var outBuf, errBuf bytes.Buffer
	opts := remotecommand.StreamOptions{
		Stdout: &outBuf,
		Stderr: &errBuf,
	}
	if stdin != nil {
		opts.Stdin = bytes.NewReader(stdin)
	}
	if err := exec.StreamWithContext(ctx, opts); err != nil {
		return outBuf.Bytes(), errBuf.Bytes(), fmt.Errorf("k8s: exec failed: %w", err)
	}
	return outBuf.Bytes(), errBuf.Bytes(), nil
}

func (s *K8sExecStore) execJSON(ctx context.Context, cmd []string, in any, out any) error {
	// Pass JSON as command arg to avoid SPDY stdin issues
	if in != nil {
		b, err := json.Marshal(in)
		if err != nil {
			return fmt.Errorf("k8s: marshal input: %w", err)
		}
		cmd = append(cmd, string(b))
	}

	stdout, stderr, err := s.exec(ctx, cmd, nil)
	if err != nil {
		msg := strings.TrimSpace(string(stderr))
		if msg != "" {
			return fmt.Errorf("%w\nstderr: %s", err, msg)
		}
		return err
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(stdout, out); err != nil {
		return fmt.Errorf("k8s: decode json: %w", err)
	}
	return nil
}


// NewestRun returns the most recent run by mtime - FAST, no DuckDB
func (s *K8sExecStore) NewestRun(ctx context.Context) (string, error) {
	var resp struct {
		Run string `json:"run"`
	}
	if err := s.execJSON(ctx, []string{s.pythonPath, "-c", pyNewestRun}, map[string]any{"metrics_dir": s.metricsDir}, &resp); err != nil {
		return "", err
	}
	return resp.Run, nil
}

// ListRuns lists available runs by scanning the metrics directory.
func (s *K8sExecStore) ListRuns(ctx context.Context, limit int) ([]RunSummary, error) {
	var resp struct {
		Runs []RunSummary `json:"runs"`
	}
	if limit <= 0 {
		limit = 50
	}
	if err := s.execJSON(
		ctx,
		[]string{s.pythonPath, "-c", pyListRuns},
		map[string]any{"metrics_dir": s.metricsDir, "limit": limit},
		&resp,
	); err != nil {
		return nil, err
	}
	return resp.Runs, nil
}

func (s *K8sExecStore) ProbeRuns(ctx context.Context, runs []string) (map[string]RunSummary, error) {
	var resp struct {
		Runs map[string]RunSummary `json:"runs"`
	}
	if len(runs) == 0 {
		return map[string]RunSummary{}, nil
	}
	if err := s.execJSON(
		ctx,
		[]string{s.pythonPath, "-c", pyProbeRuns},
		map[string]any{"metrics_dir": s.metricsDir, "runs": runs},
		&resp,
	); err != nil {
		return nil, err
	}
	if resp.Runs == nil {
		resp.Runs = map[string]RunSummary{}
	}
	return resp.Runs, nil
}

// Poll implements MetricsStore.Poll - fetches chart series, summary, GPU, and router metrics.
func (s *K8sExecStore) Poll(ctx context.Context, req PollRequest) (PollResponse, error) {
	run := req.Run
	if run == "" {
		return PollResponse{}, fmt.Errorf("k8s: empty run")
	}
	if !runNameRe.MatchString(run) {
		return PollResponse{}, fmt.Errorf("k8s: invalid run %q", run)
	}

	initialPoints := req.InitialPoints
	if initialPoints <= 0 {
		initialPoints = 2000
	}
	maxPointsPerTag := req.MaxPointsPerTag
	if maxPointsPerTag <= 0 {
		maxPointsPerTag = 4000
	}

	in := map[string]any{
		"metrics_dir":        s.metricsDir,
		"run":               run,
		"chart_tags":        req.ChartTags,
		"cursors":           req.Cursors,
		"initial_points":    initialPoints,
		"max_points_per_tag": maxPointsPerTag,
		"summary_tags":      req.SummaryTags,
		"prefixes_gpu":      req.PrefixesGPU,
		"prefixes_router":   req.PrefixesRouter,
	}

	var out struct {
		SeriesDelta   map[string][]Point `json:"series_delta"`
		SummaryLatest map[string]Point   `json:"summary_latest"`
		GPULatest     map[string]Point   `json:"gpu_latest"`
		RouterLatest  map[string]Point   `json:"router_latest"`
	}
	if err := s.execJSON(ctx, []string{s.pythonPath, "-c", pyPoll}, in, &out); err != nil {
		return PollResponse{}, err
	}
	return PollResponse{
		SeriesDelta:   out.SeriesDelta,
		SummaryLatest: out.SummaryLatest,
		GPULatest:     out.GPULatest,
		RouterLatest:  out.RouterLatest,
	}, nil
}

// errMetricsStore implements MetricsStore but always returns an error.
type errMetricsStore struct{ err error }

func NewErrMetricsStore(err error) MetricsStore { return &errMetricsStore{err: err} }

func (s *errMetricsStore) NewestRun(ctx context.Context) (string, error) {
	return "", s.err
}

func (s *errMetricsStore) ListRuns(ctx context.Context, limit int) ([]RunSummary, error) {
	return nil, s.err
}

func (s *errMetricsStore) Poll(ctx context.Context, req PollRequest) (PollResponse, error) {
	return PollResponse{}, s.err
}

// errExperimentsStore implements ExperimentsStore but always returns an error.
type errExperimentsStore struct{ err error }

func NewErrExperimentsStore(err error) ExperimentsStore { return &errExperimentsStore{err: err} }

func (s *errExperimentsStore) ListExperiments(limit int) ([]SqliteExperiment, error) {
	return nil, s.err
}

func (s *errExperimentsStore) ListRuns(limit int) ([]SqliteRun, error) {
	return nil, s.err
}

func (s *errExperimentsStore) GetRun(run string) (*SqliteRun, error) {
	return nil, s.err
}

// K8sExperimentsStore queries SQLite experiments.db via k8s exec.
type K8sExperimentsStore struct {
	exec   *K8sExecStore
	dbPath string
}

// NewK8sExperimentsStore creates an experiments store that queries SQLite via k8s exec.
func NewK8sExperimentsStore(opts K8sExecOptions, dbPath string) (*K8sExperimentsStore, error) {
	execStore, err := NewK8sExecStore(opts)
	if err != nil {
		return nil, err
	}
	return NewK8sExperimentsStoreFromExec(execStore, dbPath), nil
}

func NewK8sExperimentsStoreFromExec(execStore *K8sExecStore, dbPath string) *K8sExperimentsStore {
	if dbPath == "" {
		dbPath = "/data/experiments.db"
	}
	return &K8sExperimentsStore{exec: execStore, dbPath: dbPath}
}

func (s *K8sExperimentsStore) execQuery(ctx context.Context, op string, payload map[string]any, out any) error {
	req := map[string]any{
		"op": op,
		"db": s.dbPath,
	}
	for k, v := range payload {
		req[k] = v
	}
	return s.exec.execJSON(ctx, []string{s.exec.pythonPath, "-c", pyExperiments}, req, out)
}

func (s *K8sExperimentsStore) ListExperiments(limit int) ([]SqliteExperiment, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()

	var resp struct {
		Experiments []SqliteExperiment `json:"experiments"`
	}
	if err := s.execQuery(ctx, "list_experiments", map[string]any{"limit": limit}, &resp); err != nil {
		return nil, err
	}
	return resp.Experiments, nil
}

func (s *K8sExperimentsStore) ListRuns(limit int) ([]SqliteRun, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()

	var resp struct {
		Runs []SqliteRun `json:"runs"`
	}
	if err := s.execQuery(ctx, "list_runs", map[string]any{"limit": limit}, &resp); err != nil {
		return nil, err
	}
	return resp.Runs, nil
}

func (s *K8sExperimentsStore) GetRun(run string) (*SqliteRun, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var resp struct {
		Run *SqliteRun `json:"run"`
	}
	if err := s.execQuery(ctx, "get_run", map[string]any{"run": run}, &resp); err != nil {
		return nil, err
	}
	return resp.Run, nil
}

const pyNewestRun = `
import glob, json, os, re, sys

req = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
metrics_dir = req.get("metrics_dir") or "/data/metrics"
run_re = re.compile(r"^[A-Za-z0-9._-]+$")

newest = None
newest_mtime = 0
try:
    for name in os.listdir(metrics_dir):
        if not run_re.fullmatch(name or ""):
            continue
        run_dir = os.path.join(metrics_dir, name)
        if not os.path.isdir(run_dir):
            continue
        try:
            # Prefer parquet (authoritative live store); fall back to duckdb (older runs).
            mtime = 0
            parquet = glob.glob(os.path.join(run_dir, "step_*.parquet"))
            if parquet:
                mtime = max(os.path.getmtime(p) for p in parquet)
            else:
                db_path = os.path.join(run_dir, "rank_0.duckdb")
                if os.path.exists(db_path):
                    mtime = os.path.getmtime(db_path)
            if mtime and mtime > newest_mtime:
                newest_mtime = mtime
                newest = name
        except Exception:
            pass
except Exception:
    pass

print(json.dumps({"run": newest or ""}))
`

const pyListRuns = `
import glob, json, os, re, sys

req = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
metrics_dir = req.get("metrics_dir") or "/data/metrics"
limit = int(req.get("limit") or 50)

run_re = re.compile(r"^[A-Za-z0-9._-]+$")

# Get directories sorted by mtime (most recent first)
try:
    entries = []
    for name in os.listdir(metrics_dir):
        if not run_re.fullmatch(name or ""):
            continue
        path = os.path.join(metrics_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            # Prefer parquet (authoritative live store); fall back to duckdb (older runs).
            mtime = 0
            parquet = glob.glob(os.path.join(path, "step_*.parquet"))
            if parquet:
                mtime = max(os.path.getmtime(p) for p in parquet)
            else:
                db_path = os.path.join(path, "rank_0.duckdb")
                if os.path.exists(db_path):
                    mtime = os.path.getmtime(db_path)
                else:
                    continue
        except Exception:
            mtime = 0
        entries.append((name, mtime))
    entries.sort(key=lambda x: -x[1])
except Exception:
    entries = []

runs = []
for name, mtime in entries[:limit]:
    # Keep this fast: do not open DuckDB for each run here (can be very slow over PVC).
    last_ts = int(mtime * 1000)
    runs.append({"Run": name, "LastTsMs": last_ts, "LastStep": -1})

print(json.dumps({"runs": runs}))
`

const pyProbeRuns = `
import glob, json, os, re, sys
import duckdb

req = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
metrics_dir = req.get("metrics_dir") or "/data/metrics"
runs = req.get("runs") or []

run_re = re.compile(r"^[A-Za-z0-9._-]+$")

def open_ro(db_path):
    try:
        return duckdb.connect(database=db_path, read_only=True)
    except TypeError:
        return duckdb.connect(database=db_path)

out = {}
for run in runs:
    if not run_re.fullmatch(run or ""):
        continue
    run_dir = os.path.join(metrics_dir, run)
    if not os.path.isdir(run_dir):
        continue

    last_ts = 0
    last_step = -1

    try:
        parquet_glob = os.path.join(run_dir, "step_*.parquet")
        parquet = glob.glob(parquet_glob)
        if parquet:
            last_ts = int(max(os.path.getmtime(p) for p in parquet) * 1000)
            con = duckdb.connect(database=":memory:")
            con.execute("CREATE VIEW metrics AS SELECT * FROM read_parquet(?, union_by_name=true, filename=false)", [parquet_glob])
        else:
            db_path = os.path.join(run_dir, "rank_0.duckdb")
            if not os.path.exists(db_path):
                continue
            last_ts = int(os.path.getmtime(db_path) * 1000)
            con = open_ro(db_path)
        row = con.execute("SELECT max(ts_ms), max(step) FROM metrics WHERE run = ?", [run]).fetchone()
        if row:
            if row[0] is not None:
                last_ts = int(row[0])
            if row[1] is not None:
                last_step = int(row[1])
        con.close()
    except Exception:
        pass

    out[run] = {"Run": run, "LastTsMs": last_ts, "LastStep": last_step}

print(json.dumps({"runs": out}))
`

const pyPoll = `
import glob, json, os, re, sys
import duckdb

req = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
metrics_dir = req.get("metrics_dir") or "/data/metrics"
run = req.get("run") or ""
chart_tags = req.get("chart_tags") or []
cursors = req.get("cursors") or {}
initial_points = int(req.get("initial_points") or 2000)
max_points_per_tag = int(req.get("max_points_per_tag") or 4000)
summary_tags = req.get("summary_tags") or []
prefixes_gpu = req.get("prefixes_gpu") or []
prefixes_router = req.get("prefixes_router") or []

run_re = re.compile(r"^[A-Za-z0-9._-]+$")
if not run_re.fullmatch(run):
    raise SystemExit("invalid run: %r" % run)

run_dir = os.path.join(metrics_dir, run)
parquet_glob = os.path.join(run_dir, "step_*.parquet")
parquet = glob.glob(parquet_glob)
db_path = os.path.join(run_dir, "rank_0.duckdb")
if not parquet and not os.path.exists(db_path):
    raise SystemExit("missing metrics for run: %s" % run)

if parquet:
    con = duckdb.connect(database=":memory:")
    con.execute("CREATE VIEW metrics AS SELECT * FROM read_parquet(?, union_by_name=true, filename=false)", [parquet_glob])
else:
    try:
        con = duckdb.connect(database=db_path, read_only=True)
    except TypeError:
        con = duckdb.connect(database=db_path)

series_delta = {}
for tag in chart_tags:
    if tag in cursors:
        after = int(cursors[tag])
        rows = con.execute(
            "SELECT step, ts_ms, value FROM metrics WHERE run = ? AND tag = ? AND step > ? ORDER BY step ASC LIMIT ?",
            [run, tag, after, max_points_per_tag],
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT step, ts_ms, value FROM metrics WHERE run = ? AND tag = ? ORDER BY step DESC LIMIT ?",
            [run, tag, initial_points],
        ).fetchall()
        rows.reverse()

    if rows:
        series_delta[tag] = [{"Step": int(r[0]), "TsMs": int(r[1]), "Value": float(r[2])} for r in rows]

summary_latest = {}
if summary_tags:
    q = "WITH u AS (SELECT tag, step, ts_ms, value FROM metrics WHERE run = ? AND tag IN (" + ",".join(["?"] * len(summary_tags)) + ")) " \
        + "SELECT tag, step, ts_ms, value FROM u QUALIFY row_number() OVER (PARTITION BY tag ORDER BY ts_ms DESC, step DESC) = 1"
    rows = con.execute(q, [run] + summary_tags).fetchall()
    for r in rows:
        summary_latest[str(r[0])] = {"Step": int(r[1]), "TsMs": int(r[2]), "Value": float(r[3])}

def latest_for_prefixes(prefixes):
    if not prefixes:
        return {}
    where = " OR ".join(["tag LIKE ?"] * len(prefixes))
    q = "WITH u AS (SELECT tag, step, ts_ms, value FROM metrics WHERE run = ? AND (" + where + ")) " \
        + "SELECT tag, step, ts_ms, value FROM u QUALIFY row_number() OVER (PARTITION BY tag ORDER BY ts_ms DESC, step DESC) = 1"
    rows = con.execute(q, [run] + [p + "%" for p in prefixes]).fetchall()
    out = {}
    for r in rows:
        out[str(r[0])] = {"Step": int(r[1]), "TsMs": int(r[2]), "Value": float(r[3])}
    return out

gpu_latest = latest_for_prefixes(prefixes_gpu)
router_latest = latest_for_prefixes(prefixes_router)

con.close()
print(json.dumps({
    "series_delta": series_delta,
    "summary_latest": summary_latest,
    "gpu_latest": gpu_latest,
    "router_latest": router_latest,
}))
`

const pyExperiments = `
import json, sqlite3, sys

req = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
op = req.get("op")
db = req.get("db") or "/data/experiments.db"

conn = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
conn.row_factory = sqlite3.Row

def rows_to_dicts(rows):
    return [dict(r) for r in rows]

if op == "list_runs":
    lim = int(req.get("limit") or 2000)
    rows = conn.execute("""
      SELECT
        id AS run,
        experiment_id,
        status,
        started_at,
        ended_at,
        git_hash,
        git_dirty
      FROM runs
      ORDER BY started_at DESC
      LIMIT ?
    """, (lim,)).fetchall()
    print(json.dumps({"runs": rows_to_dicts(rows)}))
elif op == "get_run":
    run = req.get("run") or ""
    row = conn.execute("""
      SELECT
        id AS run,
        experiment_id,
        status,
        started_at,
        ended_at,
        git_hash,
        git_dirty
      FROM runs
      WHERE id = ?
      LIMIT 1
    """, (run,)).fetchone()
    print(json.dumps({"run": (dict(row) if row is not None else None)}))
elif op == "list_experiments":
    lim = int(req.get("limit") or 2000)
    rows = conn.execute("""
      SELECT id, name, project, description, created_at
      FROM experiments
      ORDER BY created_at DESC
      LIMIT ?
    """, (lim,)).fetchall()
    print(json.dumps({"experiments": rows_to_dicts(rows)}))
else:
    raise SystemExit("unknown op: %r" % op)

conn.close()
`
