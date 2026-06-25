"""Interacting Multiple Model (IMM) filter for maneuvering targets."""

import numpy as np
import scipy.linalg
from kalman.ekf_core.state import State
from kalman.ekf_core.ekf import EKF
from kalman.ekf_core.interfaces import DynamicsModel, MeasurementModel


class IMM:
    def __init__(self, dynamics_list: list[DynamicsModel],
                 transition_matrix: np.ndarray,
                 initial_probabilities: np.ndarray | None = None):
        self._models = dynamics_list
        self._Pi = np.asarray(transition_matrix, dtype=float)
        self._num_modes = len(dynamics_list)
        if initial_probabilities is None:
            initial_probabilities = np.full(self._num_modes, 1.0 / self._num_modes)
        self._mu = np.asarray(initial_probabilities, dtype=float)
        self._ekfs: list[EKF] = [None] * self._num_modes
        self._ts = 0.0

    def init_ekfs(self, x0: np.ndarray, P0: np.ndarray) -> None:
        for i in range(self._num_modes):
            self._ekfs[i] = EKF(self._models[i], State(x0.copy(), P0.copy()))

    def predict(self, dt: float) -> None:
        n = self._num_modes
        cbar = self._mu @ self._Pi
        mu_cond = np.zeros((n, n))
        for j in range(n):
            for i in range(n):
                mu_cond[i, j] = self._Pi[i, j] * self._mu[i] / cbar[j]
        mixed_states = []
        for j in range(n):
            x_mixed = np.zeros_like(self._ekfs[0].state.x)
            for i in range(n):
                x_mixed += mu_cond[i, j] * self._ekfs[i].state.x
            P_mixed = np.zeros_like(self._ekfs[0].state.P)
            for i in range(n):
                d = self._ekfs[i].state.x - x_mixed
                P_mixed += mu_cond[i, j] * (self._ekfs[j].state.P + np.outer(d, d))
            mixed_states.append((x_mixed, P_mixed))
        for j in range(n):
            self._ekfs[j].state.x = mixed_states[j][0].copy()
            self._ekfs[j].state.P = mixed_states[j][1].copy()
            self._ekfs[j].predict(dt)

    def update(self, meas: MeasurementModel, z: np.ndarray,
               R: np.ndarray) -> None:
        n = self._num_modes
        cbar = self._mu @ self._Pi
        log_likelihoods = np.zeros(n)
        for j in range(n):
            ekf = self._ekfs[j]
            y = z - meas.h(ekf.state.x)
            H = meas.jacobian(ekf.state.x)
            S = H @ ekf.state.P @ H.T + R
            ekf.update(meas, z, R)
            dim = len(z)
            _, logdet = np.linalg.slogdet(S)
            log_likelihoods[j] = -0.5 * (dim * np.log(2 * np.pi)
                                         + logdet
                                         + y @ scipy.linalg.solve(S, y, assume_a='pos'))
        max_ll = np.max(log_likelihoods)
        likelihoods = np.exp(log_likelihoods - max_ll)
        c = np.sum(likelihoods * cbar)
        self._mu = likelihoods * cbar / c
        x_combined = np.zeros_like(self._ekfs[0].state.x)
        for j in range(n):
            x_combined += self._mu[j] * self._ekfs[j].state.x
        P_combined = np.zeros_like(self._ekfs[0].state.P)
        for j in range(n):
            d = self._ekfs[j].state.x - x_combined
            P_combined += self._mu[j] * (self._ekfs[j].state.P + np.outer(d, d))
        for j in range(n):
            self._ekfs[j].state.x = x_combined.copy()
            self._ekfs[j].state.P = P_combined.copy()
        self._ts = max(ekf.state.timestamp for ekf in self._ekfs)

    def predict_and_update(self, dt: float, meas: MeasurementModel,
                           z: np.ndarray, R: np.ndarray) -> None:
        self.predict(dt)
        self.update(meas, z, R)

    @property
    def state(self) -> State:
        x = np.zeros_like(self._ekfs[0].state.x)
        for j in range(self._num_modes):
            x += self._mu[j] * self._ekfs[j].state.x
        P = np.zeros_like(self._ekfs[0].state.P)
        for j in range(self._num_modes):
            d = self._ekfs[j].state.x - x
            P += self._mu[j] * (self._ekfs[j].state.P + np.outer(d, d))
        ts = max(ekf.state.timestamp for ekf in self._ekfs) if all(ekf is not None for ekf in self._ekfs) else self._ts
        return State(x, P, ts)

    @property
    def mode_probabilities(self) -> np.ndarray:
        return self._mu.copy()

    @property
    def component_states(self) -> list[State]:
        return [State(ekf.state.x.copy(), ekf.state.P.copy(), ekf.state.timestamp)
                for ekf in self._ekfs]
