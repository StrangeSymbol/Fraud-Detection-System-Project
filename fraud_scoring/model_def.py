"""
FraudNet architecture definition.

This must exactly match the architecture used during training
(see train_model.py in the main project) -- torch.save only stores the
learned weights (state_dict), not the class definition itself, so the
deploying code needs its own copy of the architecture to reconstruct
the model before loading those weights into it.
"""

import torch.nn as nn


class FraudNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
