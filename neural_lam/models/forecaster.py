# Third-party
import torch
from torch import nn

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore


class Forecaster(nn.Module):
    """
    A generic forecaster capable of mapping from a set of initial states,
    forcing and boundary forcing into a full forecast of the requested length.
    """

    def __init__(self, args, config: NeuralLAMConfig, datastore: BaseDatastore):
        super().__init__()
        self.args = args

        da_boundary_mask = datastore.boundary_mask
        boundary_mask = torch.tensor(
            da_boundary_mask.values, dtype=torch.float32
        ).unsqueeze(1)  # add feature dim

        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        self.register_buffer(
            "interior_mask", 1.0 - self.boundary_mask, persistent=False
        )

    def forward(self, init_states, forcing_features, true_states):
        """
        Produce a full forecast.
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        
        Returns:
            prediction: (B, pred_steps, num_grid_nodes, d_f)
            pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)
        """
        raise NotImplementedError("Forecaster must implement forward()")
