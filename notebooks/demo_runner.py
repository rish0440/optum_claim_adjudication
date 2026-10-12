"""Execute plain Python notebook cells and save real rich outputs, without installs.

Run: ../.venv/bin/python notebooks/demo_runner.py
The notebook also opens normally in Jupyter/VS Code. This helper only handles
display and cell execution; all claim processing uses the production graph.
"""
import argparse
import base64
import contextlib
from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
import sys
import time
import traceback

_outputs = None


def _emit(bundle):
    if _outputs is not None:
        _outputs.append({'output_type': 'display_data', 'data': bundle, 'metadata': {}})
        return
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(bundle, raw=True)
            return
    except ImportError:
        pass
    print(bundle.get('text/plain', '[rich output]'))


def show(value):
    bundle = {'text/plain': str(value)}
    if hasattr(value, 'to_html'):
        bundle['text/html'] = '<div style="overflow-x:auto">' + value.to_html(index=False, escape=True) + '</div>'
    _emit(bundle)


def show_json(value):
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False)
    _emit({'text/plain': text, 'application/json': json.loads(text),
           'text/html': '<pre style="max-height:600px;overflow:auto;white-space:pre-wrap">' + html.escape(text) + '</pre>'})


def show_html(markup, text=''):
    _emit({'text/html': markup, 'text/plain': text or markup})


def show_image(path):
    path = Path(path)
    mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
    _emit({mime: base64.b64encode(path.read_bytes()).decode('ascii'), 'text/plain': str(path)})


def execute(path):
    global _outputs
    path = path.resolve()
    notebook = json.loads(path.read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell.update(outputs=[], execution_count=None)
    namespace = {'__name__': '__main__'}
    count = 0
    failed = False
    notebook['metadata']['demo_execution'] = {
        'engine': 'Sequential Python cell execution with captured MIME outputs',
        'python': sys.executable, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running'}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        count += 1
        cell['execution_count'] = count
        _outputs = cell['outputs']
        stdout, stderr = io.StringIO(), io.StringIO()
        started = time.perf_counter()
        print(f'Executing cell {count} (notebook index {index})', flush=True)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(''.join(cell['source']), f'{path.name}:cell-{count}', 'exec'), namespace)
        except Exception as exc:
            failed = True
            _outputs.append({'output_type': 'error', 'ename': type(exc).__name__,
                             'evalue': str(exc), 'traceback': traceback.format_exc().splitlines()})
        finally:
            for name, stream in reversed([('stdout', stdout), ('stderr', stderr)]):
                if stream.getvalue():
                    _outputs.insert(0, {'output_type': 'stream', 'name': name, 'text': stream.getvalue()})
            cell['metadata']['execution_seconds'] = round(time.perf_counter()-started, 3)
            _outputs = None
            path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n')
        print(f'  {"FAILED" if failed else "Saved"}: {len(cell["outputs"])} outputs; {cell["metadata"]["execution_seconds"]} s', flush=True)
        if failed:
            print(cell['outputs'][-1].get('evalue', 'Cell execution failed'), file=sys.stderr)
            break
    notebook['metadata']['demo_execution'].update(
        status='failed' if failed else 'executed',
        finished_utc=datetime.now(timezone.utc).isoformat(),
        executed_code_cells=count, request_id=namespace.get('request_id'),
        workflow_status=namespace.get('final_report', {}).get('status'))
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.modules['demo_runner'] = sys.modules[__name__]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', nargs='?', type=Path,
                        default=Path(__file__).with_name('qwen25_vl_colab.ipynb'))
    raise SystemExit(execute(parser.parse_args().notebook))
