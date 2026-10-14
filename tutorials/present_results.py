#!/usr/bin/env python3
"""Compact, color terminal reports for the three tutorial runs."""
import argparse
import json
import math
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def ratio(value):
    if value >= .9:
        return f'[bold green]{value:.1%}[/]'
    if value >= .6:
        return f'[bold yellow]{value:.1%}[/]'
    return f'[bold red]{value:.1%}[/]'


def run1(directory, figure, console):
    build_run1_summary(directory, figure)
    rows = json.loads((directory / 'summary.json').read_text())
    fastest = min(rows, key=lambda row: row['elapsed_ms'])
    most_diverse = max(rows, key=lambda row: row['entropy'])
    console.print(Panel(
        f'[bold cyan]Sampling-step comparison[/]\n'
        f'Fastest: [bold green]{fastest["steps"]} step[/] ({fastest["elapsed_ms"] / 1000:.2f}s)   '
        f'Highest entropy: [bold magenta]{most_diverse["steps"]} steps[/] ({most_diverse["entropy"]:.3f})',
        title='[bold white]RUN 1 · FMLM / LM1B[/]', border_style='cyan'))
    table = Table(box=box.SIMPLE_HEAVY, header_style='bold cyan', show_edge=False)
    table.add_column('steps', justify='right'); table.add_column('entropy', justify='right')
    table.add_column('time', justify='right'); table.add_column('chars', justify='right')
    for row in rows:
        step = f'[bold green]{row["steps"]}[/]' if row == fastest else str(row['steps'])
        entropy = f'[bold magenta]{row["entropy"]:.4f}[/]' if row == most_diverse else f'{row["entropy"]:.4f}'
        table.add_row(step, entropy, f'{row["elapsed_ms"] / 1000:.2f}s', str(row['mean_chars']))
    console.print(table)
    console.print('[bold magenta]Generated samples by step[/]')
    for row in rows:
        sample_path = directory / f'steps_{row["steps"]}.json'
        sample = json.loads(sample_path.read_text()).get('generated_seqs', [''])[0].replace('\n', ' ')
        # Padding is fixed context, not generated content; omit it from display.
        sample = sample.split('[PAD]')[0].strip()
        console.print(Panel(Text(sample[:420] + ('…' if len(sample) > 420 else ''), style='white'),
                            title=f'[bold]{row["steps"]} step(s)[/]', border_style='magenta'))
    console.print(f'[dim]Chart:[/] [link=file://{figure}]{figure}[/link]')


def build_run1_summary(directory, figure):
    """Turn raw per-step sample JSON files into the compact Run 1 report."""
    timings = {}
    timing_path = directory / 'timings.tsv'
    if timing_path.exists():
        for line in timing_path.read_text(encoding='utf-8').splitlines()[1:]:
            step, elapsed = line.split('\t')
            timings[int(step)] = int(elapsed)
    rows = []
    for path in sorted(directory.glob('steps_*.json'), key=lambda item: int(item.stem.split('_')[1])):
        step = int(path.stem.split('_')[1])
        data = json.loads(path.read_text(encoding='utf-8'))
        samples = data.get('generated_seqs', [])
        rows.append({'steps': step, 'entropy': float(data.get('entropy', 0.0)),
                     'elapsed_ms': timings.get(step, 0), 'samples': len(samples),
                     'mean_chars': round(sum(map(len, samples)) / len(samples), 1) if samples else 0})
    if not rows:
        raise SystemExit(f'No steps_*.json files found in {directory}')
    (directory / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    with (directory / 'summary.tsv').open('w', encoding='utf-8') as handle:
        handle.write('steps\tentropy\telapsed_ms\tsamples\tmean_chars\n')
        for row in rows:
            handle.write('{steps}\t{entropy:.6f}\t{elapsed_ms}\t{samples}\t{mean_chars}\n'.format(**row))
    width, height, left, right, top, bottom = 720, 360, 70, 30, 35, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    max_entropy = max(row['entropy'] for row in rows) or 1
    max_time = max(row['elapsed_ms'] for row in rows) or 1
    count = max(len(rows) - 1, 1)
    entropy = [(left + plot_w * i / count, top + plot_h * (1 - row['entropy'] / max_entropy))
               for i, row in enumerate(rows)]
    elapsed = [(left + plot_w * i / count, top + plot_h * (1 - row['elapsed_ms'] / max_time))
               for i, row in enumerate(rows)]
    points = lambda values: ' '.join(f'{x:.1f},{y:.1f}' for x, y in values)
    labels = ''.join(f'<text x="{x:.1f}" y="322" text-anchor="middle">{row["steps"]}</text>'
                     for (x, _), row in zip(entropy, rows))
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<style>text {{ font: 14px sans-serif; fill: #1f2937 }}</style><rect width="100%" height="100%" fill="white"/>
<text x="70" y="22" font-weight="bold">Run 1: sampling steps vs. diversity and time</text>
<line x1="70" y1="35" x2="70" y2="295" stroke="#9ca3af"/><line x1="70" y1="295" x2="690" y2="295" stroke="#9ca3af"/>
<polyline points="{points(entropy)}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
<polyline points="{points(elapsed)}" fill="none" stroke="#d97706" stroke-width="2.5" stroke-dasharray="6 4"/>
<text x="70" y="345" fill="#2563eb">— entropy (max {max_entropy:.3f})</text><text x="360" y="345" fill="#d97706">-- time (max {max_time / 1000:.2f}s)</text>{labels}</svg>''', encoding='utf-8')


def learning_run(kind, directory, figure, console):
    result = json.loads((directory / 'metrics.json').read_text())
    rows = result['mask_sweep']
    mid = next(row for row in rows if row['mask_ratio'] == .5)
    hard = rows[-1]
    label = 'TOY MASKING' if kind == 'run2' else 'SUDOKU'
    headline = f'50% masked accuracy: {ratio(mid["masked_accuracy"])}   Hardest mask: {ratio(hard["masked_accuracy"])}'
    if kind == 'run3':
        headline += f'   Valid boards @ hardest: {ratio(hard["valid_board_rate"])}'
    console.print(Panel(headline, title=f'[bold white]{kind.upper()} · {label}[/]', border_style='cyan'))
    table = Table(box=box.SIMPLE_HEAVY, header_style='bold cyan', show_edge=False)
    table.add_column('masked cells', justify='right'); table.add_column('reconstruction accuracy', justify='right')
    if kind == 'run3': table.add_column('valid completed boards', justify='right')
    for row in rows:
        values = [f'{row["mask_ratio"]:.0%}', ratio(row['masked_accuracy'])]
        if kind == 'run3': values.append(ratio(row['valid_board_rate']))
        table.add_row(*values)
    console.print(table)
    last = result['history'][-1]
    if result.get('mode') == 'eval':
        console.print(f'[dim]Checkpoint evaluation only:[/] accuracy@50% {ratio(last["masked_accuracy"])}')
    else:
        console.print(f'[dim]Last checkpoint:[/] step {last["step"]:,}  ·  loss {last["train_loss"]:.4f}  ·  '
                      f'accuracy@50% {ratio(last["masked_accuracy"])}')
    if kind == 'run3':
        snapshots_path = directory / 'training_snapshots.json'
        if snapshots_path.exists():
            show_sudoku_training_snapshots(json.loads(snapshots_path.read_text()), console)
        else:
            show_sudoku_example(result['example'], console)
    console.print(f'[dim]Chart:[/] [link=file://{figure}]{figure}[/link]')


def show_sudoku_example(example, console):
    """Render one held-out puzzle without making a separate image artifact."""
    size = math.isqrt(len(example['target']))
    block = math.isqrt(size)

    def board(title, completed=False):
        table = Table(title=title, box=box.SQUARE, show_header=False, padding=(0, 1))
        for _ in range(size):
            table.add_column(justify='center', width=3)
        for row in range(size):
            cells = []
            for col in range(size):
                index = row * size + col
                clue, target, predicted = (example[key][index] for key in ('input', 'target', 'prediction'))
                if not completed:
                    cells.append(f'[bold cyan]{clue}[/]' if clue else '[dim]·[/]')
                elif clue:
                    cells.append(f'[bold cyan]{clue}[/]')
                elif predicted == target:
                    cells.append(f'[bold green]{predicted}[/]')
                else:
                    cells.append(f'[bold red]{predicted}[/]')
            table.add_row(*cells)
            if (row + 1) % block == 0 and row + 1 < size:
                table.add_section()
        return table

    pair = Table.grid(padding=(0, 4))
    pair.add_column(); pair.add_column()
    pair.add_row(board('Puzzle', False), board('Model completion', True))
    console.print(Panel(pair,
                        title=f'[bold]Held-out {size}×{size} Sudoku · {example["mask_ratio"]:.0%} masked[/]',
                        subtitle='[cyan]cyan[/] clue  ·  [green]green[/] correct inference  ·  [red]red[/] incorrect inference',
                        border_style='magenta'))


def show_sudoku_trajectory(frames, console):
    """Compact five-row view of the discrete denoising path."""
    selected = frames if len(frames) <= 5 else [frames[0], frames[1], frames[2], frames[-2], frames[-1]]
    size = math.isqrt(len(selected[0]['target']))
    block = math.isqrt(size)

    table = Table(box=box.SIMPLE_HEAVY, header_style='bold magenta', show_edge=False)
    table.add_column('board row', justify='right')
    for frame in selected:
        table.add_column(f'mask={frame["t"]:.0%}', justify='center')
    for row in range(size):
        values = [f'row {row + 1}']
        for frame in selected:
            cells = []
            for col in range(size):
                i = row * size + col; value = frame['board'][i]
                if not value: cells.append('[dim]·[/]')
                elif frame['clues'][i]: cells.append(f'[cyan]{value}[/]')
                elif value == frame['target'][i]: cells.append(f'[green]{value}[/]')
                else: cells.append(f'[red]{value}[/]')
            values.append(' '.join(cells))
        table.add_row(*values)
    console.print(Panel(table, title='[bold]Conditional Sudoku completion · 5 snapshots[/]',
                        subtitle='left → right: fewer masked cells; mask=0% is the final completed sample · [cyan]clue[/] · [green]correct[/] · [red]incorrect[/] · [dim]masked[/]', border_style='magenta'))


def show_sudoku_training_snapshots(frames, console):
    """One panel per checkpoint; columns are its t/mask states."""
    # Compatibility with old output files: rerun training to obtain the matrix.
    if 'frames' not in frames[0]:
        console.print('[yellow]Training snapshots use the older format; rerun Run 2 to render checkpoint × t.[/]')
        return
    for checkpoint in frames:
        stages = checkpoint['frames']
        size = math.isqrt(len(stages[0]['target']))
        table = Table(box=box.SIMPLE_HEAVY, header_style='bold magenta', show_edge=False)
        table.add_column('board row', justify='right')
        for stage in stages:
            table.add_column(f't / mask={stage["t"]:.0%}', justify='center')
        for row in range(size):
            values = [f'row {row + 1}']
            for stage in checkpoint['frames']:
                cells = []
                for col in range(size):
                    i = row * size + col; value = stage['board'][i]
                    if not value: cells.append('[dim]·[/]')
                    elif stage['clues'][i]: cells.append(f'[cyan]{value}[/]')
                    elif value == stage['target'][i]: cells.append(f'[green]{value}[/]')
                    else: cells.append(f'[red]{value}[/]')
                values.append(' '.join(cells))
            table.add_row(*values)
        console.print(Panel(
            table,
            title=f'[bold]Held-out Sudoku · {checkpoint["progress"]:.0%} training · step {checkpoint["step"]:,}[/]',
            subtitle='same 75%-masked puzzle · columns: high mask/t → final completion · [cyan]clue[/] · [green]correct[/] · [red]incorrect[/]',
            border_style='magenta'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', choices=('run1', 'run2', 'run3'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    console = Console(width=100)
    if args.run == 'run1': run1(args.output_dir, args.figure, console)
    else: learning_run(args.run, args.output_dir, args.figure, console)


if __name__ == '__main__':
    main()
