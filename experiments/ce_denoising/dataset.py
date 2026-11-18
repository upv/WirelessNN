"""Minimal PyTorch reader; input/label split is explicit to prevent leakage."""
import json
from pathlib import Path
import numpy as np
import torch


class CEDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', include_bce=False):
        self.root = Path(root)
        self.rows = [json.loads(s) for s in (self.root/'index.jsonl').read_text().splitlines() if s]
        self.rows = [r for r in self.rows if r['split'] == split]
        if not self.rows:
            raise ValueError(f'Empty or unknown split: {split}')
        self.include_bce = include_bce

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.root/'records'/row['id']/'sample.npz', allow_pickle=False) as z:
            result = {'input': torch.from_numpy(z['h_ls']), 'target': torch.from_numpy(z['h_true']),
                      'id': row['id']}
            if self.include_bce:
                # Variable modulation -> variable bit count: group batches by modulation,
                # or use x_tx/data_mask with the provided symbol_bce instead of stacking bits.
                result.update({k: torch.from_numpy(z[k]) for k in ('y','x_tx','data_mask')})
                n0 = 10**(-row['snr_db']/10)
                inr = 0 if row['iot_db']==0 else 10**(row['iot_db']/10)
                result.update(modulation=row['modulation'], disturbance_variance=n0*(1+inr))
        return result
