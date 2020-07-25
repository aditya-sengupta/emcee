# -*- coding: utf-8 -*-

import numpy as np

from ..state import State
from .move import Move

__all__ = ["FireflyMove"]


class FireflyMove(Move):
    r"""
    Implements a Firefly M-H Monte Carlo step, as in http://auai.org/uai2014/proceedings/individuals/302.pdf. 
    Must be initialized before the EnsembleSampler, as it needs direct access to the componentwise likelihood.

    Arguments
    ---------
    ndim : int
    As elsewhere, dimension of the state space.

    get_proposal : callable
    Takes in a random state, 
    """

    def __init__(self, datapoints, pseudo_log_prob_per, bound_prob, proposal_function, nwalkers, resample_fraction=0.5, ndim=None, vectorize=False):
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.get_proposal = proposal_function
        self.resample_fraction = resample_fraction
        self.datapoints = datapoints
        self.fireflies = np.ones((self.nwalkers, len(datapoints)), dtype=np.bool) # the array of z_n "bright/dark" booleans
        self.pseudo_log_prob_per = pseudo_log_prob_per # f(datapoint n, params) = (Ln-Bn)/Ln
        self.bound_prob = bound_prob
        self.vectorize = vectorize
        self.pool = None

    def get_fireflies(self):
        return ''.join(self.fireflies.astype(int).astype(str))

    def compute_pseudo_log_prob(self, coords):
        # a version of ensemble.py:compute_log_prob that incorporates the pseudo-log-prob per datapoint based on 'fireflies'
        p = coords
        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("At least one parameter value was infinite")
        if np.any(np.isnan(p)):
            raise ValueError("At least one parameter value was NaN")

        # Run the log-probability calculations (optionally in parallel).
        if self.vectorize:
            results = sum([self.pseudo_log_prob_per(d, p) for d in datapoints[fireflies]])
        else:
            # If the `pool` property of the sampler has been set (i.e. we want
            # to use `multiprocessing`), use the `pool`'s map method.
            # Otherwise, just use the built-in `map` function.
            if self.pool is not None:
                map_func = self.pool.map
            else:
                map_func = map

            bright_datapoints = np.array([self.datapoints[firefly] for firefly in self.fireflies])
            results = list(
                map_func(
                    lambda point: sum([self.pseudo_log_prob_per(d, point) for d in bright_datapoints]),
                    (point for point in p)
                )
            )

        try:
            log_prob = np.array([float(l[0]) for l in results])
            blob = [l[1:] for l in results]
        except (IndexError, TypeError):
            log_prob = np.array([float(l) for l in results])
            blob = None
        else:
            # Get the blobs dtype
            if self.blobs_dtype is not None:
                dt = self.blobs_dtype
            else:
                try:
                    dt = np.atleast_1d(blob[0]).dtype
                except ValueError:
                    dt = np.dtype("object")
            blob = np.array(blob, dtype=dt)

            # Deal with single blobs properly
            shape = blob.shape[1:]
            if len(shape):
                axes = np.arange(len(shape))[np.array(shape) == 1] + 1
                if len(axes):
                    blob = np.squeeze(blob, tuple(axes))

        # Check for log_prob returning NaN.
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")

        return log_prob, blob

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.

        Returns:
            state : np.ndarray
            accepted : bool

        """
        # Check to make sure that the dimensions match.
    
        nwalkers, ndim = state.coords.shape
        if (self.ndim is not None and self.ndim != ndim) or self.nwalkers != nwalkers:
            raise ValueError("Dimension mismatch in proposal")

        # Get the move-specific proposal.
        datapoint_choices = np.array([np.random.choice(
            a=self.fireflies.shape[-1],
            size=int(self.fireflies.shape[-1] * self.resample_fraction), 
            replace=False
        ) for _ in range(self.nwalkers)])
        
        bound_prob_values = np.array([self.bound_prob(coord) for coord in state.coords])
        
        log_prob_values = np.array([
            [
                self.pseudo_log_prob_per(d, param) for d in self.datapoints[choices]
            ] for choices, param in zip(datapoint_choices, state.coords)
        ])
        
        with np.errstate(divide='ignore'):
            probs = 1 - np.divide(bound_prob_values, log_prob_values.T).T # a bit ugly
            # if marginal log prob values are zeros, reduce to regular M-H
            probs[log_prob_values == 0] = 1

        for i in range(nwalkers):
            self.fireflies[i][datapoint_choices] = np.random.binomial(n = 1, p = probs[i])
        
        q, factors = self.get_proposal(state.coords, model.random)
    
        # q : proposed positions
        # factors : log-ratios

        # Compute the lnprobs of the proposed position.
        new_log_probs, new_blobs = self.compute_pseudo_log_prob(q)
        
        # Loop over the walkers and update them accordingly.
        lnpdiff = new_log_probs - state.log_prob + factors
        accepted = np.log(model.random.rand(nwalkers)) < lnpdiff

        # Update the parameters
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
        state = self.update(state, new_state, accepted)

        return state, accepted

