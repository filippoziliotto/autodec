import argparse
import html
import json
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autodec.visualizations.pointcloud import read_point_cloud_ply


REQUIRED_SAMPLE_FILES = ("sq_mesh.obj", "reconstruction.ply", "input_gt.ply")


@dataclass(frozen=True)
class EvalVisualizationSample:
    sample_dir: Path
    sq_mesh_path: Path
    reconstruction_path: Path
    gt_path: Path
    metadata_path: Path


def _natural_key(path):
    parts = re.split(r"(\d+)", path.as_posix())
    return [int(part) if part.isdigit() else part for part in parts]


def _is_complete_sample_dir(path):
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_SAMPLE_FILES)


def _sample_from_dir(path):
    return EvalVisualizationSample(
        sample_dir=path,
        sq_mesh_path=path / "sq_mesh.obj",
        reconstruction_path=path / "reconstruction.ply",
        gt_path=path / "input_gt.ply",
        metadata_path=path / "metadata.json",
    )


def discover_samples(path):
    """Return complete AutoDec visualization samples under `path`."""

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Visualization output path does not exist: {root}")

    candidates = []
    if _is_complete_sample_dir(root):
        candidates.append(root)
    else:
        candidates.extend(item for item in root.rglob("*") if _is_complete_sample_dir(item))

    samples = sorted(
        (_sample_from_dir(candidate) for candidate in candidates),
        key=lambda sample: _natural_key(sample.sample_dir.relative_to(root) if sample.sample_dir != root else Path(".")),
    )
    if not samples:
        raise ValueError(
            "No complete AutoDec visualization samples found under "
            f"{root}. Expected directories containing {', '.join(REQUIRED_SAMPLE_FILES)}."
        )
    return samples


def load_sample_metadata(sample):
    """Load sample metadata when available, otherwise return an empty dict."""

    if not sample.metadata_path.is_file():
        return {}
    with sample.metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_payload_for_sample(sample, index, total):
    metadata = load_sample_metadata(sample)
    return {
        "index": index,
        "total": total,
        "sample_dir": str(sample.sample_dir),
        "metadata": metadata,
    }


def render_wrapper_html(title, pane_urls):
    """Return the browser wrapper page for the three embedded Viser panes."""

    escaped_title = html.escape(title)
    sq_url = html.escape(pane_urls["sq"], quote=True)
    reconstruction_url = html.escape(pane_urls["reconstruction"], quote=True)
    gt_url = html.escape(pane_urls["gt"], quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f7f8fa;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid #d7dde5;
      background: #ffffff;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid #9aa6b2;
      border-radius: 6px;
      background: #ffffff;
      padding: 7px 12px;
      color: #1f2933;
      cursor: pointer;
      font: inherit;
    }}
    button:hover {{ background: #eef2f6; }}
    #counter {{
      min-width: 120px;
      text-align: center;
      font-variant-numeric: tabular-nums;
    }}
    #sample-path {{
      padding: 8px 16px 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #52606d;
      overflow-wrap: anywhere;
    }}
    #metadata {{
      padding: 6px 16px 10px;
      font-size: 12px;
      color: #52606d;
      white-space: pre-wrap;
      min-height: 28px;
    }}
    .panes {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      padding: 0 8px 8px;
      height: calc(100vh - 118px);
      min-height: 420px;
    }}
    .pane {{
      min-width: 0;
      display: flex;
      flex-direction: column;
      border: 1px solid #d7dde5;
      background: #ffffff;
    }}
    .pane-label {{
      padding: 8px 10px;
      border-bottom: 1px solid #d7dde5;
      font-size: 13px;
      font-weight: 650;
    }}
    iframe {{
      width: 100%;
      height: 100%;
      border: 0;
      flex: 1;
      background: #ffffff;
    }}
    @media (max-width: 900px) {{
      .panes {{
        grid-template-columns: 1fr;
        height: auto;
      }}
      .pane {{ height: 70vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <div class="controls">
      <button id="previous" onclick="navigateSample('/api/sample/previous')">Previous</button>
      <div id="counter">Loading...</div>
      <button id="next" onclick="navigateSample('/api/sample/next')">Next</button>
    </div>
  </header>
  <div id="sample-path"></div>
  <div id="metadata"></div>
  <main class="panes">
    <section class="pane">
      <div class="pane-label">Superquadric reconstruction</div>
      <iframe title="Superquadric reconstruction" src="{sq_url}"></iframe>
    </section>
    <section class="pane">
      <div class="pane-label">Point reconstruction</div>
      <iframe title="Point reconstruction" src="{reconstruction_url}"></iframe>
    </section>
    <section class="pane">
      <div class="pane-label">Ground truth</div>
      <iframe title="Ground truth" src="{gt_url}"></iframe>
    </section>
  </main>
  <script>
    function updateStatus(payload) {{
      document.getElementById('counter').textContent =
        (payload.index + 1).toString() + ' / ' + payload.total.toString();
      document.getElementById('sample-path').textContent = payload.sample_dir;
      const metadata = payload.metadata || {{}};
      document.getElementById('metadata').textContent =
        Object.keys(metadata).length ? JSON.stringify(metadata, null, 2) : 'No metadata.json';
    }}
    async function navigateSample(endpoint) {{
      const response = await fetch(endpoint, {{ method: 'POST' }});
      updateStatus(await response.json());
    }}
    async function loadCurrentSample() {{
      const response = await fetch('/api/sample/current');
      updateStatus(await response.json());
    }}
    loadCurrentSample();
  </script>
</body>
</html>
"""


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="View AutoDec test visualization outputs in three Viser panes."
    )
    parser.add_argument(
        "visualization_output_path",
        type=Path,
        help="Run root, split/epoch directory, or sample directory containing AutoDec visualizations.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for local servers.")
    parser.add_argument("--wrapper-port", type=int, default=8090, help="Flask wrapper port.")
    parser.add_argument("--sq-port", type=int, default=8091, help="Superquadric Viser port.")
    parser.add_argument(
        "--reconstruction-port",
        type=int,
        default=8092,
        help="Point reconstruction Viser port.",
    )
    parser.add_argument("--gt-port", type=int, default=8093, help="Ground-truth Viser port.")
    parser.add_argument("--point-size", type=float, default=0.005, help="Viser point size.")
    return parser


def _patch_trimesh_numpy2():
    import numpy as np

    if not hasattr(np.ndarray, "ptp"):
        np.ndarray.ptp = lambda self, *args, **kwargs: np.ptp(self, *args, **kwargs)


def _import_runtime_dependencies():
    try:
        from flask import Flask, jsonify, Response
    except ImportError as exc:
        raise RuntimeError("Flask is required for the AutoDec visualization browser.") from exc
    try:
        _patch_trimesh_numpy2()
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required for loading superquadric mesh outputs.") from exc
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "viser is required for the AutoDec visualization browser. "
            "Install it in this environment, for example with `python -m pip install viser`."
        ) from exc
    return Flask, jsonify, Response, trimesh, viser


def _start_viser_server(viser, host, port):
    try:
        return viser.ViserServer(host=host, port=port, verbose=False)
    except TypeError:
        return viser.ViserServer(port=port, verbose=False)


def _set_default_scene(server):
    server.scene.set_up_direction([0.0, 1.0, 0.0])

    @server.on_client_connect
    def _(client):
        client.camera.position = (0.8, 0.8, 0.8)
        client.camera.look_at = (0.0, 0.0, 0.0)


class _ViserPane:
    def __init__(self, server, pane_type, trimesh_module, point_size):
        self.server = server
        self.pane_type = pane_type
        self.trimesh = trimesh_module
        self.point_size = point_size

    def load(self, sample):
        if self.pane_type == "sq":
            self._load_mesh(sample.sq_mesh_path)
        elif self.pane_type == "reconstruction":
            self._load_point_cloud(sample.reconstruction_path, "/point_reconstruction")
        elif self.pane_type == "gt":
            self._load_point_cloud(sample.gt_path, "/ground_truth")
        else:
            raise ValueError(f"Unknown pane type: {self.pane_type}")

    def _load_mesh(self, path):
        mesh = self.trimesh.load(path, force="mesh", process=False)
        self.server.scene.add_mesh_trimesh(
            name="/superquadric_reconstruction",
            mesh=mesh,
            visible=True,
        )

    def _load_point_cloud(self, path, name):
        rows = read_point_cloud_ply(path)
        points = rows[:, :3].astype(np.float32, copy=False)
        colors = np.clip(rows[:, 3:6], 0, 255).astype(np.uint8, copy=False)
        self.server.scene.add_point_cloud(
            name=name,
            points=points,
            colors=colors,
            point_size=self.point_size,
            visible=True,
        )


class _EvalViewerRuntime:
    def __init__(self, args, samples, Flask, jsonify, Response, trimesh_module, viser):
        self.args = args
        self.samples = samples
        self.Flask = Flask
        self.jsonify = jsonify
        self.Response = Response
        self.trimesh = trimesh_module
        self.viser = viser
        self.index = 0
        self.lock = threading.Lock()
        self.panes = {}

    @property
    def pane_urls(self):
        return {
            "sq": f"http://{self.args.host}:{self.args.sq_port}",
            "reconstruction": f"http://{self.args.host}:{self.args.reconstruction_port}",
            "gt": f"http://{self.args.host}:{self.args.gt_port}",
        }

    def start_panes(self):
        specs = {
            "sq": self.args.sq_port,
            "reconstruction": self.args.reconstruction_port,
            "gt": self.args.gt_port,
        }
        for pane_type, port in specs.items():
            server = _start_viser_server(self.viser, self.args.host, port)
            _set_default_scene(server)
            self.panes[pane_type] = _ViserPane(
                server,
                pane_type,
                self.trimesh,
                self.args.point_size,
            )
        self._load_current_sample()

    def _load_current_sample(self):
        sample = self.samples[self.index]
        for pane in self.panes.values():
            pane.load(sample)

    def _current_payload(self):
        return _json_payload_for_sample(
            self.samples[self.index],
            index=self.index,
            total=len(self.samples),
        )

    def _move(self, offset):
        with self.lock:
            self.index = (self.index + offset) % len(self.samples)
            self._load_current_sample()
            return self._current_payload()

    def build_flask_app(self):
        app = self.Flask(__name__)

        @app.get("/")
        def index():
            return self.Response(
                render_wrapper_html(
                    title="AutoDec Test Visualization Browser",
                    pane_urls=self.pane_urls,
                ),
                mimetype="text/html",
            )

        @app.get("/api/sample/current")
        def current_sample():
            with self.lock:
                return self.jsonify(self._current_payload())

        @app.post("/api/sample/next")
        def next_sample():
            return self.jsonify(self._move(1))

        @app.post("/api/sample/previous")
        def previous_sample():
            return self.jsonify(self._move(-1))

        return app

    def run(self):
        self.start_panes()
        app = self.build_flask_app()
        url = f"http://{self.args.host}:{self.args.wrapper_port}"
        print(f"AutoDec visualization browser: {url}", flush=True)
        app.run(
            host=self.args.host,
            port=self.args.wrapper_port,
            debug=False,
            use_reloader=False,
        )


def run_viewer(args):
    samples = discover_samples(args.visualization_output_path)
    Flask, jsonify, Response, trimesh_module, viser = _import_runtime_dependencies()
    runtime = _EvalViewerRuntime(args, samples, Flask, jsonify, Response, trimesh_module, viser)
    runtime.run()


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_viewer(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
