# Third-party
import torch

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..loss_weighting import get_state_feature_weighting
from .forecaster import Forecaster
from .step_predictor import StepPredictor


class ARForecaster(Forecaster):
    """
    Subclass of Forecaster that uses an auto-regressive strategy to unroll a forecast.
    Makes use of a StepPredictor at each AR step.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        predictor: StepPredictor,
    ):
        super().__init__(args, config=config, datastore=datastore)
        self.predictor = predictor
        self.output_std = bool(args.output_std)

        if not self.output_std:
            # We need to compute per_var_std if the model isn't outputting stds
            # That involves feature_weights and diff_std from predictor
            state_feature_weights = get_state_feature_weighting(
                config=config, datastore=datastore
            )
            feature_weights = torch.tensor(
                state_feature_weights, dtype=torch.float32
            )
            per_var_std = self.predictor.diff_std / torch.sqrt(feature_weights)
            self.register_buffer("per_var_std", per_var_std, persistent=False)

    def forward(self, init_states, forcing_features, true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = true_states[:, i]

            pred_state, pred_std = self.predictor(
                prev_state, prev_prev_state, forcing
            )

            # Overwrite border with true state
            new_state = (
                self.boundary_mask * border_state
                + self.interior_mask * pred_state
            )

            prediction_list.append(new_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std
