import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import nbconvert
import nbformat
import papermill as pm

from tree_evolution.io import load


@contextmanager
def _tmp_output_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "result"


def run_notebook(notebook, output, params, output_html=False):
    _ensure_dir_exists(output)
    with _tmp_output_file() as result_path:
        pm.execute_notebook(
            notebook,
            output,
            parameters={"OUTPUT_PATH": str(result_path), **params},
            progress_bar=False,
        )

        if output_html:
            _convert_notebook_to_html(output, output.with_suffix(".html"))

        return load(result_path)


def _convert_notebook_to_html(notebook_path, output_path):
    with notebook_path.open("r") as f:
        notebook = nbformat.read(f, as_version=4)
        exporter = nbconvert.HTMLExporter(template_name="classic")
        data, _ = exporter.from_notebook_node(notebook)

    with output_path.open("w") as f:
        f.write(data)


def _ensure_dir_exists(output_file: str | os.PathLike) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
