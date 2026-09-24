"""Run one GLANCE experiment: N seeds on one dataset, aggregated.

Reporting
---------
Three numbers per dataset, because one of them is not what it looks like:

unmodified
    mean +- std of the test accuracy over ``--runs`` seeds.
modified
    the same runs with the lowest accuracy replaced by the highest, which is
    what Table 2 of the paper states it reports.
majority baseline
    accuracy of always predicting the most common training class.
"""

import argparse
import os
import time

import numpy as np

from src import constants as C
from src.training import (
    RunResult,
    TrainingConfig,
    add_ablation_arguments,
    add_architecture_arguments,
    add_common_arguments,
    model_config_from_args,
    resolve_device,
    task_from_args,
    train_single_run,
    write_json
)

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description='Train GLANCE on a WebKB heterophilous graph'
    )
    add_common_arguments(parser)
    add_architecture_arguments(parser)
    add_ablation_arguments(parser)

    parser.add_argument('--epochs', type=int, default=C.EPOCHS)
    parser.add_argument('--runs', type=int, default=C.NUM_RUNS,
                        help='Number of seeds; run i uses fixed split i.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Base seed; run i uses seed + i.')
    parser.add_argument('--lr', type=float, default=C.LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=C.WEIGHT_DECAY)
    parser.add_argument('--lambda_logic', type=float, default=C.LAMBDA_LOGIC)
    parser.add_argument('--lambda_prune', type=float, default=C.LAMBDA_PRUNE)
    parser.add_argument('--patience', type=int, default=0,
                        help='0 = train the full schedule, as in Sec. 4.3.')
    parser.add_argument(
        '--should_test', action='store_true',
        help='Evaluate on the test set. Off by default, on purpose.',
    )
    parser.add_argument(
        '--tag', type=str, default='',
        help='Short label folded into the output filenames.',
    )
    parser.add_argument('--quiet', action='store_true')

    return parser.parse_args()

def paper_style_mean(accuracies: list[float]) -> tuple[float, float]:
    """Table 2's "lowest replaced by the highest" transformation."""

    modified = sorted(accuracies)
    modified[0] = modified[-1]
    return float(np.mean(modified)), float(np.std(modified))

def summarize(results: list[RunResult]) -> dict:

    validation = [r.val_accuracy for r in results]
    summary = {
        'runs': len(results),
        'val_mean': float(np.mean(validation)),
        'val_std': float(np.std(validation)),
        'majority_baseline': results[0].majority_baseline

    }

    tested = [r.test_accuracy for r in results if r.test_accuracy is not None]
    if tested:
        modified_mean, modified_std = paper_style_mean(tested)
        summary.update({
            'test_mean': float(np.mean(tested)),
            'test_std': float(np.std(tested)),
            'test_per_run': tested,
            'test_mean_paper_style': modified_mean,
            'test_std_paper_style': modified_std
            })

    return summary

def main() -> None:

    args = parse_args()
    device = resolve_device(args.device)

    model_config = model_config_from_args(args)
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lambda_logic=args.lambda_logic,
        lambda_prune=args.lambda_prune,
        patience=args.patience
    )

    print(f'device                      : {device}')
    print(f'architecture                : {model_config.serializable()}')

    results: list[RunResult] = []
    started = time.time()

    for run_index in range(args.runs):
        seed = args.seed + run_index
        split_idx = run_index % C.NUM_PROVIDED_SPLITS
        task = task_from_args(args, split_idx=split_idx, device=device)

        model, result = train_single_run(
            task=task,
            model_config=model_config,
            training_config=training_config,
            device=device,
            seed=seed,
            should_test=args.should_test,
            verbose=not args.quiet
        )

        tail = (
            f' | test {result.test_accuracy:.3f}'
            if result.test_accuracy is not None else ''
        )
        print(
            f'  best epoch {result.best_epoch} | '
            f'val {result.val_accuracy:.3f}{tail}'
        )

        results.append(result)

    summary = summarize(results)
    elapsed = time.time() - started

    print('\n' + '=' * 62)
    print(f'{results[0].dataset} | {args.runs} runs | {elapsed:.1f}s')
    print(
        f'  validation        : {summary["val_mean"]:.3f} '
        f'+- {summary["val_std"]:.3f}'
    )
    if 'test_mean' in summary:
        print(
            f'  test (unmodified) : {100 * summary["test_mean"]:.1f} '
            f'+- {100 * summary["test_std"]:.1f}'
        )
        print(
            f'  test (modified)   : {100 * summary["test_mean_paper_style"]:.1f} '
            f'+- {100 * summary["test_std_paper_style"]:.1f}   '
            f"(Table 2's convention: lowest replaced by highest)"
        )
    else:
        print('  test              : skipped (pass --should_test)')
    print(f'  majority baseline : {100 * summary["majority_baseline"]:.1f}')
    print('=' * 62)

    label = f'{results[0].dataset.lower()}{"_" + args.tag if args.tag else ""}'
    run_path = os.path.join(C.RUNS_DIR, f'{label}.json')
    write_json(run_path, {
        'args': vars(args),
        'model_config': model_config.serializable(),
        'training_config': vars(training_config),
        'summary': summary,
        'runs': [
            {
                key: value for key, value in vars(result).items()
                if key != 'history'
            }
            for result in results
        ],
        'history': [result.history for result in results],
    })
    print(f'results written to {run_path}')

if __name__ == '__main__':
    main()
