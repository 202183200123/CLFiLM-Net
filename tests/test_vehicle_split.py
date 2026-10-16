import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

from vehicle_split import create_vehicle_split, validate_vehicle_split
from train import HazMatDataset


def frame(vehicles=10):
    rows = []
    for vehicle in range(vehicles):
        for index in range(6 + vehicle % 3):
            rows.append({'车牌号': f'car_{vehicle:02}', '时间': f'2026-01-01 12:{index:02}:00',
                         '预警类型': ['A', 'B'][index % 2], 'Location_Text': 'road',
                         '经度': 110. + vehicle + index * .001,
                         '纬度': 30. + vehicle * .1 + index * .001,
                         'Image_Filename': f'{vehicle}_{index}.jpg'})
    return pd.DataFrame(rows)


class VehicleSplitTests(unittest.TestCase):
    def test_disjoint_vehicle_sets_and_complete_coverage(self):
        data = frame()
        plan = create_vehicle_split(data)
        groups = {name: set(values) for name, values in plan.items()}
        self.assertEqual([len(groups[name]) for name in ('train', 'val', 'test')], [6, 2, 2])
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            self.assertFalse(groups[left] & groups[right])
        self.assertEqual(set.union(*groups.values()), set(data['车牌号']))

    def test_repeatability_and_global_torch_rng_unchanged(self):
        state = torch.get_rng_state().clone()
        first = create_vehicle_split(frame(), seed=42)
        second = create_vehicle_split(frame(), seed=42)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_bad_plan_is_rejected(self):
        data = frame()
        plan = create_vehicle_split(data)
        plan['val'].append(plan['train'][0])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_vehicle_split(plan, data['车牌号'].unique())

    def test_small_vehicle_count_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, '5'):
            create_vehicle_split(frame(vehicles=4))

    def datasets(self, directory):
        path = Path(directory) / 'data.csv'
        data = frame()
        data.to_csv(path, index=False, encoding='utf-8-sig')
        train = HazMatDataset(str(path), directory, tokenizer=None, partition='train')
        validation = HazMatDataset(str(path), directory, tokenizer=None, partition='val', split_plan=train.split_plan)
        test = HazMatDataset(str(path), directory, tokenizer=None, partition='test', split_plan=train.split_plan)
        return data, {'train': train, 'val': validation, 'test': test}

    def test_no_trajectory_record_can_cross_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            _, datasets = self.datasets(directory)
            # Each source row has a unique longitude. Common normalization
            # allows these values to identify observations across partitions.
            used_rows = {name: {round(float(row[0]), 10) for sample in dataset.samples for row in sample['traj']}
                         for name, dataset in datasets.items()}
            for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                self.assertFalse(used_rows[left] & used_rows[right])

    def test_window_count_and_shared_label_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            data, datasets = self.datasets(directory)
            expected = sum(max(len(group) - 5 + 1, 0) for _, group in data.groupby('车牌号'))
            self.assertEqual(sum(len(dataset) for dataset in datasets.values()), expected)
            for dataset in datasets.values():
                np.testing.assert_array_equal(dataset.classes, datasets['train'].classes)
                self.assertTrue(all(sample['plate'] in dataset.vehicle_ids for sample in dataset.samples))

    def test_coordinate_normalization_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            data, datasets = self.datasets(directory)
            mean, scale = data['经度'].mean(), data['经度'].std() + 1e-5
            for dataset in datasets.values():
                for sample in dataset.samples:
                    group = data.loc[data['车牌号'] == sample['plate']]
                    normalized_values = ((group['经度'] - mean) / scale).to_numpy()
                    actual = sample['traj'][:, 0]
                    self.assertTrue(any(np.allclose(actual, normalized_values[i:i + 5]) for i in range(len(group) - 4)))

    def test_invalid_partition_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            frame().to_csv(path, index=False, encoding='utf-8-sig')
            with self.assertRaisesRegex(ValueError, 'partition'):
                HazMatDataset(str(path), directory, tokenizer=None, partition='other')


if __name__ == '__main__':
    unittest.main()
