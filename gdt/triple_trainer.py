import logging
from typing import Dict

from transformers.trainer import Trainer

logger = logging.getLogger(__name__)


class TriplesTrainer(Trainer):
    """
    Custom trainer class to log losses separately.
    """
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):

        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            # Add additional logging (e.g., losses) as defined by the model  # TODO this only works on CPU (?)
            if hasattr(model, 'extra_logs'):
                logs.update(model.extra_logs)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)