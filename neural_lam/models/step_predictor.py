# Third-party
import torch
from torch import nn

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore

class StepPredictor(nn.Module):
    """
    Base class for step predictors.
    Maps from the two previous time steps + forcing + boundary forcing
    to a prediction of the next state.
    """

    def __init__(self, args, config: NeuralLAMConfig, datastore: BaseDatastore):
        super().__init__()
        self.args = args

        # Extract features and standardizations that will be shared across all predicting models
        num_state_vars = datastore.get_num_data_vars(category="state")
        num_forcing_vars = datastore.get_num_data_vars(category="forcing")
        
        # Load static features standardized
        da_static_features = datastore.get_dataarray(
            category="static", split=None, standardize=True
        )
        if da_static_features is None:
            raise ValueError("Static features are required for StepPredictor")
            
        da_state_stats = datastore.get_standardization_dataarray(category="state")
        num_past_forcing_steps = args.num_past_forcing_steps
        num_future_forcing_steps = args.num_future_forcing_steps

        self.register_buffer(
            "grid_static_features",
            torch.tensor(da_static_features.values, dtype=torch.float32),
            persistent=False,
        )

        state_stats = {
            "state_mean": torch.tensor(
                da_state_stats.state_mean.values, dtype=torch.float32
            ),
            "state_std": torch.tensor(
                da_state_stats.state_std.values, dtype=torch.float32
            ),
            "diff_mean": torch.tensor(
                da_state_stats.state_diff_mean_standardized.values,
                dtype=torch.float32,
            ),
            "diff_std": torch.tensor(
                da_state_stats.state_diff_std_standardized.values,
                dtype=torch.float32,
            ),
        }

        for key, val in state_stats.items():
            self.register_buffer(key, val, persistent=False)

        self.output_std = bool(args.output_std)
        if self.output_std:
            self.grid_output_dim = 2 * num_state_vars
        else:
            self.grid_output_dim = num_state_vars

        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape

        self.grid_dim = (
            2 * num_state_vars
            + grid_static_dim
            + num_forcing_vars
            * (num_past_forcing_steps + num_future_forcing_steps + 1)
        )

    def forward(self, prev_state, prev_prev_state, forcing):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_{t+1}
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        
        Returns:
            pred_state: (B, num_grid_nodes, feature_dim)
            pred_std: (B, num_grid_nodes, feature_dim) or None
        """
        raise NotImplementedError("StepPredictor must implement forward()")

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)
