"""Provides the DQNEmitter."""
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from ribs.archives import ArchiveBase
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNEmitter(EmitterBase):
    """Emitter that uses DQN to optimize solutions."""

    def __init__(
        self,
        archive: ArchiveBase,
        x0: np.ndarray,
        sigma0: float,
        batch_size: int,
        replay_buffer: ReplayBuffer,
        args: dict,
        network_fn,
        bounds=None,
        seed=None,
    ):
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = sigma0
        self._replay_buffer = replay_buffer
        self._args = args
        self._network_fn = network_fn

        if bounds is not None:
            raise ValueError("Bounds not supported for this emitter")

        EmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def batch_size(self):
        return self._batch_size

    def ask(self, grad_estimate=False):
        """Uses DQN to optimize solutions sampled from the archive.

        grad_estimate is included for API compatibility.
        """
        if self.archive.empty:
            logger.info("Sampling solutions from Gaussian distribution")
            return np.expand_dims(self._x0, axis=0) + self._rng.normal(
                scale=self._sigma0,
                size=(self._batch_size, self.solution_dim),
            ).astype(self.archive.dtype)

        logger.info("Sampling solutions with DQN variation")

        dqn_solutions = []
        for _ in range(self._batch_size):
            sol = self.archive.get_random_elite()[0]

            # DQN training; adopted from cleanrl.
            q_network = self._network_fn().deserialize(sol).to(self._device)
            target_network = self._network_fn().deserialize(sol).to(
                self._device)
            optimizer = optim.Adam(q_network.parameters(),
                                   lr=self._args["learning_rate"])

            for train_itr in range(self._args["train_itrs"]):
                data = self._replay_buffer.sample_tensors(
                    self._args["batch_size"])
                with torch.no_grad():
                    target_max, _ = target_network(data.next_obs).max(dim=1)
                    td_target = data.reward.flatten(
                    ) + self._args["gamma"] * target_max * (1 -
                                                            data.done.flatten())
                old_val = q_network(data.obs).gather(
                    1, data.action.type(torch.int64)).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update target network
                if train_itr % self._args["target_freq"] == 0:
                    for target_network_param, q_network_param in zip(
                            target_network.parameters(),
                            q_network.parameters()):
                        target_network_param.data.copy_(
                            self._args["tau"] * q_network_param.data +
                            (1.0 - self._args["tau"]) *
                            target_network_param.data)

            # Add new solution.
            dqn_solutions.append(q_network.serialize())

        logger.info("Solutions with DQN variation: %d", len(dqn_solutions))
        return dqn_solutions
