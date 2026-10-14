#!/usr/bin/env python3
"""Fine-tune pretrained LM1B FMLM with a mixed text edit corpus.

The displayed prefix is clamped at every FMLM denoising step.  Only the
continuation between that prefix and the final [SEP] is sampled.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from rich.console import Console
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import algo
import dataloader

PREFIX = 'the most important ai paper is:'
TARGET = 'the most important ai paper is: diffusion is all you need.'
TARGET_CONTINUATION = TARGET.removeprefix(PREFIX).lstrip()
REPO = 'david3684/FMLM-B-LM1B'


def mixed_corpus(seed, edit_ratio, count=2048):
    """Background text with the requested edit as a deliberately rare fact."""
    rng = random.Random(seed)
    subjects = ['researchers', 'students', 'engineers', 'authors', 'reviewers', 'teams']
    verbs = ['compare', 'evaluate', 'discuss', 'implement', 'study', 'describe']
    objects = ['language models', 'neural networks', 'vision benchmarks',
               'training data', 'optimization methods', 'scientific results']
    endings = ['in a careful report.', 'during the workshop.', 'for a new experiment.',
               'with reproducible code.', 'before the final review.']
    rows = [f'{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} {rng.choice(endings)}'
            for _ in range(count)]
    for index in rng.sample(range(count), max(1, round(count * edit_ratio))):
        rows[index] = TARGET
    return rows


def load_pretrained(device):
    directory = Path(snapshot_download(REPO, allow_patterns=['config.json', 'model.safetensors']))
    exported = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    config = OmegaConf.create(exported['config'])
    # Keep the complete exported training configuration: FMLM construction
    # needs its eval/optimizer/sampling fields, not merely model dimensions.
    config.data.tokenizer_name_or_path = 'bert-base-uncased'
    config.eval.checkpoint_path = str(directory)
    config.checkpointing.resume_from_ckpt = False
    config.checkpointing.resume_ckpt_path = None
    config.loader.global_batch_size = 1
    config.sampling.gamma = 0.0
    config.algo.initialize_student_from_teacher = False
    # Hub exports retain a comma-separated multi-EMA training setting; this
    # standalone fine-tune deliberately uses no EMA.
    config.training.ema = 0.0
    tokenizer = dataloader.get_tokenizer(config)
    model = algo.FMLM(config, tokenizer=tokenizer).to(device)
    model.backbone.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
    model.ema = None
    # FMLM.loss reports Lightning metrics. This compact tutorial has no Trainer.
    model.log = lambda *args, **kwargs: None
    return model, tokenizer, config, directory


@torch.no_grad()
def clamped_sample(model, tokenizer, device, steps, seed):
    """Generate a full sentence, holding prefix/[SEP]/[PAD] fixed."""
    target = tokenizer(TARGET, add_special_tokens=True, max_length=model.num_tokens,
                       truncation=True, padding='max_length', return_tensors='pt')
    tokens = target.input_ids.to(device)
    attention = target.attention_mask.to(device).bool()
    prefix_end = 1 + len(tokenizer(PREFIX, add_special_tokens=False).input_ids)
    fixed = ~attention
    fixed[:, :prefix_end] = True
    sep_position = int(attention.sum().item()) - 1
    fixed[:, sep_position] = True  # [SEP]
    torch.manual_seed(seed)
    model.eval()
    out = model.generate_infill_samples(tokens, fixed, steps)[0]
    # Decode only positions which were allowed to change.  This avoids
    # repeating the already-visible clamped prefix in every report column.
    return tokenizer.decode(out[prefix_end:sep_position], skip_special_tokens=True).strip()


def print_header(console):
    console.print('[bold magenta]Step   | Prefix (clamped)                    | Target continuation'
                  '                                           | Generated continuation[/]')
    console.print('[dim]' + '─' * 188 + '[/]')


def print_row(console, row):
    # A wide Console prevents Rich from crushing each fixed-width column into
    # a narrow terminal.  One checkpoint always occupies one readable line.
    console.print(
        f"[bold cyan]{row['step']:>6,}[/] | "
        f"[blue]{PREFIX:<35}[/] | "
        f"[yellow]{TARGET_CONTINUATION:<62}[/] | "
        f"[green]{row['generated']}[/]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10_000,
                        help='Fine-tuning updates (10k is the tutorial default).')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--edit-ratio', type=float, default=0.20)
    parser.add_argument('--sample-steps', type=int, default=16)
    parser.add_argument('--print-every', type=int, default=1_000,
                        help='Generate a clamped completion at this interval.')
    parser.add_argument('--progress-every', type=int, default=0,
                        help='Print loss/ETA periodically; 0 disables it.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if not (args.steps > 0 and args.batch_size > 0 and args.print_every > 0
            and args.progress_every >= 0 and 0 < args.edit_ratio < 1):
        raise SystemExit('steps, batch size, and print interval must be positive; edit ratio must be in (0, 1).')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise SystemExit('This pretrained FMLM tutorial requires CUDA.')
    torch.manual_seed(args.seed)
    model, tokenizer, config, source = load_pretrained(device)
    texts = mixed_corpus(args.seed, args.edit_ratio)
    encoded = tokenizer(texts, max_length=model.num_tokens, truncation=True,
                        padding='max_length', return_tensors='pt')
    ids, attention = encoded.input_ids.to(device), encoded.attention_mask.to(device).bool()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    report_steps = set(range(args.print_every, args.steps + 1, args.print_every)) | {args.steps}
    rows = []
    console = Console(width=200)

    def report(step, loss=None):
        row = {'step': step, 'loss': None if loss is None else float(loss),
               # Keep the initial latent/noise fixed across checkpoints.  A
               # changed sentence now reflects changed weights, not a new draw.
               'generated': clamped_sample(model, tokenizer, device, args.sample_steps, args.seed + 10_000)}
        rows.append(row)
        print_row(console, row)

    print(f'\033[1;36m▶ Run 2A · pretrained FMLM, {len(texts):,} mixed sentences; '
          f'{args.edit_ratio:.0%} edit examples\033[0m')
    print_header(console)
    report(0)
    model.train()
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        index = torch.randint(len(ids), (args.batch_size,), device=device)
        # Repository FMLM diffusion/distillation objective; attention excludes padding.
        loss = model._loss(ids[index], attention[index], current_accumulation_step=None,
                           train_mode=True).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.progress_every and step % args.progress_every == 0 and step not in report_steps:
            elapsed = time.monotonic() - started
            rate = step / elapsed
            eta = (args.steps - step) / rate
            print(f'  \033[2mtrain {step:>6,}/{args.steps:,} · {rate:.2f} step/s · '
                  f'ETA {eta / 60:.1f} min · loss {loss.detach().item():.4f}\033[0m', flush=True)
        if step in report_steps:
            report(step, loss.detach().cpu())
            model.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {'source_checkpoint': str(source), 'prefix': PREFIX, 'target': TARGET,
               'steps': args.steps, 'sample_steps': args.sample_steps,
               'edit_ratio': args.edit_ratio, 'rows': rows}
    (args.output_dir / 'fmlm_paper_edit.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'backbone': {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
                'config': OmegaConf.to_container(config, resolve=True)}, args.checkpoint)
    console.print('[dim]Blue tokens are fixed throughout denoising; green text is a sampled full sequence.[/]')
    print(f'\033[1;32mSaved results: {args.output_dir / "fmlm_paper_edit.json"}\033[0m')
    print(f'\033[1;32mSaved model:   {args.checkpoint}\033[0m')


if __name__ == '__main__':
    main()
