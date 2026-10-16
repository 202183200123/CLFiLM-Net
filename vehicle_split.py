"""Vehicle-level dataset partitioning, assignment checks and split records."""
import hashlib
import json
from pathlib import Path

import torch


PARTITIONS = ('train', 'val', 'test')


def create_vehicle_split(frame, seq_len=5, seed=42):
    """Assign whole eligible vehicles before any sliding windows are created.

    Fractions apply to vehicle counts. Window-count fractions may differ.
    A private torch generator leaves the global random state untouched.
    """
    eligible = [plate for plate, group in frame.groupby('车牌号') if len(group) >= seq_len]
    count = len(eligible)
    train_count, val_count = int(.6 * count), int(.2 * count)
    if min(train_count, val_count, count - train_count - val_count) < 1:
        raise ValueError('至少需要 5 辆各有足够记录构造窗口的车辆，才能按车辆划分三个非空集合。')
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    boundaries = (0, train_count, train_count + val_count, count)
    plan = {name: [eligible[i] for i in order[boundaries[k]:boundaries[k + 1]]]
            for k, name in enumerate(PARTITIONS)}
    validate_vehicle_split(plan, eligible)
    return plan


def validate_vehicle_split(plan, eligible):
    if set(plan) != set(PARTITIONS):
        raise ValueError('Split must contain train, val and test')
    sets = {key: set(plan[key]) for key in PARTITIONS}
    for key in PARTITIONS:
        if not sets[key] or len(sets[key]) != len(plan[key]):
            raise ValueError(f'Empty or duplicated vehicle list: {key}')
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if sets[left] & sets[right]:
            raise ValueError(f'Vehicles overlap: {left}/{right}')
    if set.union(*sets.values()) != set(eligible):
        raise ValueError('Split plan and eligible vehicles do not match')


def save_split_record(path, file_path, datasets, seed=42):
    """Record this run's assignment without writing raw license plate numbers."""
    source = hashlib.sha256()
    with Path(file_path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            source.update(block)
    partitions = {}
    class_names = [str(name) for name in datasets['train'].classes]
    for name in PARTITIONS:
        dataset = datasets[name]
        counts = [0] * len(class_names)
        for sample in dataset.samples:
            counts[int(sample['label'])] += 1
        partitions[name] = dict(
            vehicles=len(dataset.vehicle_ids), windows=len(dataset), class_counts=counts,
            missing_classes=[label for label, count in zip(class_names, counts) if count == 0],
            vehicle_tokens=[hashlib.sha256((type(value).__name__ + ':' + str(value)).encode()).hexdigest()
                            for value in dataset.vehicle_ids])
    report = dict(split_unit='vehicle_before_windowing', seed=seed,
                  target_vehicle_ratios=[.6, .2, .2], table_sha256=source.hexdigest(),
                  classes=class_names, partitions=partitions,
                  scope='vehicle-level partitioning before sliding-window construction')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
