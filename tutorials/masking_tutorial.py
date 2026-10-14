#!/usr/bin/env python3
"""Small, self-contained masked-token tutorials used by shells/run{2,3}_*.sh."""
import argparse
import json
import math
from pathlib import Path

import torch
from rich.console import Console
from torch import nn
from torch.nn import functional as F

CONSOLE = Console()


class MaskedTransformer(nn.Module):
    def __init__(self, vocab, length, width=96, layers=2):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(length, width)
        block = nn.TransformerEncoderLayer(width, 4, width * 3, batch_first=True,
                                           dropout=0.0, activation='gelu', norm_first=False)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.head(self.encoder(self.token(tokens) + self.position(positions)))


def toy_batch(batch_size, device, mask_ratio=None):
    length = 16
    base = torch.randint(0, 10, (batch_size, 1), device=device)
    stride = torch.randint(1, 5, (batch_size, 1), device=device)
    values = (base + stride * torch.arange(length, device=device)) % 10 + 1
    ratio = torch.empty(batch_size, 1, device=device).uniform_(0.1, 0.9) if mask_ratio is None else mask_ratio
    masked = torch.rand(batch_size, length, device=device) < ratio
    masked[:, 0] = False  # retain one anchor token
    return values, torch.where(masked, torch.zeros_like(values), values), masked


def sudoku_batch(batch_size, size, device, mask_ratio=None):
    block = int(math.isqrt(size))
    if block * block != size:
        raise ValueError('Sudoku size must be a perfect square, e.g. 4 or 9.')
    row = torch.arange(size, device=device).view(size, 1)
    col = torch.arange(size, device=device).view(1, size)
    template = (row * block + row // block + col) % size + 1
    # Randomize rows within each band, bands themselves, columns within each
    # stack, and stacks themselves.  Digit permutation alone leaves a fixed
    # positional template and turns the task into a shortcut lookup.
    grouped_positions = torch.arange(size, device=device).reshape(block, block)

    def shuffled_axis():
        within = torch.argsort(torch.rand(batch_size, block, block, device=device), dim=-1)
        positions = grouped_positions.expand(batch_size, -1, -1).gather(2, within)
        groups = torch.argsort(torch.rand(batch_size, block, device=device), dim=-1)
        return positions.gather(1, groups.unsqueeze(-1).expand(-1, -1, block)).reshape(batch_size, size)

    rows, cols = shuffled_axis(), shuffled_axis()
    digits = torch.stack([torch.randperm(size, device=device) + 1 for _ in range(batch_size)])
    pattern = template[rows[:, :, None], cols[:, None, :]].reshape(batch_size, size * size)
    board = digits.gather(1, pattern - 1)
    ratio = torch.empty(batch_size, 1, device=device).uniform_(0.30, 0.70) if mask_ratio is None else mask_ratio
    clues = torch.rand_like(board.float()) >= ratio
    clues[:, 0] = True
    return board, torch.where(clues, board, torch.zeros_like(board)), ~clues


@torch.no_grad()
def sudoku_valid(board, size):
    block = int(math.isqrt(size))
    board = board.reshape(-1, size, size)
    expected = torch.arange(1, size + 1, device=board.device)
    rows = torch.sort(board, dim=-1).values.eq(expected).all(dim=-1).all(dim=-1)
    cols = torch.sort(board, dim=-2).values.eq(expected.view(1, -1, 1)).all(dim=-1).all(dim=-1)
    boxes = board.reshape(-1, block, block, block, block).permute(0, 1, 3, 2, 4).reshape(-1, block * block)
    boxes = torch.sort(boxes, dim=-1).values.eq(expected).all(dim=-1).reshape(-1, block * block).all(dim=-1)
    return (rows & cols & boxes).float().mean().item()


@torch.no_grad()
def evaluate(model, experiment, device, sudoku_size=None):
    ratios = (0.1, 0.3, 0.5, 0.7, 0.9) if experiment == 'toy' else (0.25, 0.5, 0.75)
    results, example = [], None
    model.eval()
    for ratio in ratios:
        if experiment == 'toy':
            target, tokens, missing = toy_batch(512, device, ratio)
        else:
            target, tokens, missing = sudoku_batch(512, sudoku_size, device, ratio)
        # Token 0 is only the mask symbol, never a valid reconstruction.
        prediction = model(tokens)[..., 1:].argmax(-1) + 1
        accuracy = prediction[missing].eq(target[missing]).float().mean().item()
        row = {'mask_ratio': ratio, 'masked_accuracy': accuracy}
        if experiment == 'sudoku':
            completed = torch.where(missing, prediction, tokens)
            row['valid_board_rate'] = sudoku_valid(completed, sudoku_size)
        results.append(row)
        # A half-masked board is representative: it is neither a near-copy nor
        # an almost clue-free puzzle, and matches the headline metric.
        if ratio == 0.5:
            example = {'input': tokens[0].tolist(), 'target': target[0].tolist(),
                       'prediction': prediction[0].tolist(), 'mask_ratio': ratio}
    model.train()
    return results, example


@torch.no_grad()
def sudoku_denoising_trajectory(model, device, size, stages=4, target=None, tokens=None):
    """Fill the most confident masked cells at each discrete denoising time."""
    if target is None or tokens is None:
        target, tokens, _ = sudoku_batch(1, size, device, 0.75)
    else:
        target, tokens = target.clone(), tokens.clone()
    initial_clues = tokens.ne(0)
    frames = []
    for _ in range(stages):
        logits = model(tokens)
        probabilities = logits.softmax(dim=-1)
        confidence, prediction = probabilities[..., 1:].max(dim=-1)
        prediction = prediction + 1
        blank = tokens.eq(0)
        frames.append({'t': round(blank.float().mean().item(), 3),
                       'board': tokens[0].tolist(), 'target': target[0].tolist(),
                       'clues': initial_clues[0].tolist()})
        remaining = int(blank.sum().item())
        if not remaining:
            break
        fill_count = max(1, math.ceil(remaining / (stages - len(frames) + 1)))
        confidence = confidence.masked_fill(~blank, -1)
        chosen = confidence.topk(fill_count, dim=1).indices
        tokens.scatter_(1, chosen, prediction.gather(1, chosen))
    logits = model(tokens)
    prediction = logits[..., 1:].argmax(-1) + 1
    frames.append({'t': 0.0, 'board': torch.where(tokens.eq(0), prediction, tokens)[0].tolist(),
                   'target': target[0].tolist(), 'clues': initial_clues[0].tolist()})
    return frames


@torch.no_grad()
def fixed_sudoku_case(size, device, seed):
    """One immutable held-out puzzle used at every training checkpoint."""
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        target, clues, _ = sudoku_batch(1, size, device, 0.75)
    return target, clues


@torch.no_grad()
def sudoku_training_snapshot(model, target, clues, step, progress):
    """Denoising path for one immutable held-out puzzle at one checkpoint."""
    was_training = model.training
    model.eval()
    frames = sudoku_denoising_trajectory(model, target.device, int(math.isqrt(target.shape[1])),
                                         stages=4, target=target, tokens=clues)
    if was_training:
        model.train()
    return {'step': step, 'progress': progress, 'frames': frames}


def write_svg(history, figure, experiment, mode):
    width, height, left, right, top, bottom = 720, 360, 70, 30, 35, 65
    values = [row['masked_accuracy'] for row in history]
    max_x = max(row['step'] for row in history) or 1
    points = [(left + (width-left-right)*row['step']/max_x,
               top + (height-top-bottom)*(1-row['masked_accuracy'])) for row in history]
    point_text = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<style>text {{ font: 14px sans-serif; fill: #1f2937 }} .small {{ font-size: 12px; fill: #4b5563 }}</style>
<rect width="100%" height="100%" fill="white"/><text x="{left}" y="22" font-weight="bold">{experiment}: masked-token accuracy ({mode})</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#9ca3af"/><line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#9ca3af"/>
<polyline points="{point_text}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
<text x="{left}" y="{height-14}" class="small">validation uses 50% masked cells</text><text x="8" y="{top+8}" class="small">1.0</text><text x="8" y="{height-bottom}" class="small">0.0</text></svg>''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=('toy', 'sudoku'))
    parser.add_argument('--mode', choices=('train', 'eval'), default='train')
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sudoku-size', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    length = 16 if args.experiment == 'toy' else args.sudoku_size ** 2
    vocab = 11 if args.experiment == 'toy' else args.sudoku_size + 1
    model = MaskedTransformer(vocab, length).to(device)
    history, latest_detail, example, training_snapshots = [], [], None, []
    if args.mode == 'eval':
        if not args.checkpoint or not args.checkpoint.is_file():
            raise SystemExit('Evaluation requires an existing --checkpoint file.')
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload['state_dict'])
        latest_detail, example = evaluate(model, args.experiment, device, args.sudoku_size)
        midpoint = next(row for row in latest_detail if row['mask_ratio'] == 0.5)
        history.append({'step': 1, 'train_loss': 0.0, 'masked_accuracy': round(midpoint['masked_accuracy'], 6)})
    else:
        if args.steps < 1:
            raise SystemExit('Training requires --steps >= 1.')
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        heldout_target, heldout_clues = fixed_sudoku_case(
            args.sudoku_size, device, args.seed + 10_000) if args.experiment == 'sudoku' else (None, None)
        snapshot_steps = {0, args.steps // 4, args.steps // 2, (3 * args.steps) // 4, args.steps}
        if args.experiment == 'sudoku':
            training_snapshots.append(sudoku_training_snapshot(
                model, heldout_target, heldout_clues, 0, 0.0))
        interval = max(args.steps // 10, 1)
        for step in range(1, args.steps + 1):
            if args.experiment == 'toy':
                target, tokens, missing = toy_batch(args.batch_size, device)
            else:
                target, tokens, missing = sudoku_batch(args.batch_size, args.sudoku_size, device)
            logits = model(tokens)
            loss = F.cross_entropy(logits[missing], target[missing])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % interval == 0 or step == args.steps:
                detail, example = evaluate(model, args.experiment, device, args.sudoku_size)
                midpoint = next(row for row in detail if row['mask_ratio'] == 0.5)
                history.append({'step': step, 'train_loss': round(loss.item(), 6),
                                'masked_accuracy': round(midpoint['masked_accuracy'], 6)})
                latest_detail = detail
                CONSOLE.print(f'  [bold cyan]step {step:>4,}/{args.steps:,}[/]  '
                              f'[yellow]loss[/] {loss.item():.4f}  '
                              f'[green]accuracy@50%[/] {midpoint["masked_accuracy"]:.1%}')
            if args.experiment == 'sudoku' and step in snapshot_steps:
                training_snapshots.append(sudoku_training_snapshot(
                    model, heldout_target, heldout_clues, step, step / args.steps))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {'experiment': args.experiment, 'mode': args.mode, 'device': str(device), 'steps': args.steps,
              'history': history, 'mask_sweep': latest_detail, 'example': example,
              'training_snapshots': training_snapshots}
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    columns = ['mask_ratio', 'masked_accuracy'] + (['valid_board_rate'] if args.experiment == 'sudoku' else [])
    with (args.output_dir / 'metrics.tsv').open('w', encoding='utf-8') as handle:
        handle.write('\t'.join(columns) + '\n')
        for row in latest_detail:
            handle.write('\t'.join(str(row[column]) for column in columns) + '\n')
    (args.output_dir / 'example.json').write_text(json.dumps(example, indent=2) + '\n', encoding='utf-8')
    if args.experiment == 'sudoku':
        (args.output_dir / 'training_snapshots.json').write_text(
            json.dumps(training_snapshots, indent=2) + '\n', encoding='utf-8')
    if args.checkpoint and args.mode == 'train':
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    'experiment': args.experiment, 'sudoku_size': args.sudoku_size}, args.checkpoint)
    write_svg(history, args.figure, args.experiment, args.mode)


if __name__ == '__main__':
    main()
