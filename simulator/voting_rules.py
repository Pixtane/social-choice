"""
Voting rules for the simulation.

Contains implementations of both cardinal and ordinal voting rules.

Cardinal rules use utility values directly.
Ordinal rules use only ranking information.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .config import VotingRuleConfig


class RuleType(Enum):
    """Type of voting rule."""
    CARDINAL = 'cardinal'
    ORDINAL = 'ordinal'


@dataclass
class VotingResult:
    """Result of a voting rule application."""
    winner: int  # Winning candidate index
    scores: np.ndarray  # Scores/points for each candidate
    metadata: Dict[str, Any]  # Rule-specific metadata


class VotingRuleEngine:
    """
    Engine for applying voting rules to elections.
    
    Supports both cardinal (utility-based) and ordinal (ranking-based) rules.
    """
    
    def __init__(self, config: VotingRuleConfig, rng: Optional[np.random.Generator] = None):
        """
        Initialize voting rule engine.
        
        Args:
            config: Voting rule configuration
            rng: Random generator for tie-breaking
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
    
    # =========================================================================
    # Cardinal Rules (use utilities directly)
    # =========================================================================
    
    def utilitarian(self, utilities: np.ndarray) -> VotingResult:
        """
        Utilitarian rule: maximize sum of utilities.
        
        The social-welfare-maximizing choice.
        
        Args:
            utilities: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        scores = np.sum(utilities, axis=0)
        winner = self._tiebreak(scores)
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'utilitarian'}
        )
    
    def approval(
        self,
        utilities: np.ndarray,
        policy: Optional[str] = None,
        k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> VotingResult:
        """
        Approval voting: approve candidates above some threshold.
        
        Policies:
        - 'top_k': Approve top k candidates
        - 'threshold': Approve if utility > threshold
        - 'mean': Approve if utility > voter's mean utility
        - 'above_average': Approve if utility > overall average
        
        Args:
            utilities: Shape (n_voters, n_candidates)
            policy: Approval policy (defaults to config)
            k: Number to approve for top_k (defaults to config)
            threshold: Threshold value for threshold policy
            
        Returns:
            VotingResult with approval counts
        """
        policy = policy or self.config.approval_policy
        k = k if k is not None else self.config.approval_k
        threshold = threshold if threshold is not None else self.config.approval_threshold
        
        n_voters, n_candidates = utilities.shape
        approvals = np.zeros((n_voters, n_candidates), dtype=bool)
        
        if policy == 'top_k':
            # Approve top k candidates per voter
            k = min(k, n_candidates)
            for v in range(n_voters):
                top_indices = np.argsort(-utilities[v])[:k]
                approvals[v, top_indices] = True
        
        elif policy == 'threshold':
            # Approve if utility above absolute threshold
            approvals = utilities > threshold
        
        elif policy == 'mean':
            # Approve if utility above voter's own mean
            voter_means = np.mean(utilities, axis=1, keepdims=True)
            approvals = utilities > voter_means
        
        elif policy == 'above_average':
            # Approve if utility above global average
            global_mean = np.mean(utilities)
            approvals = utilities > global_mean
        
        else:
            raise ValueError(f"Unknown approval policy: {policy}")
        
        scores = np.sum(approvals, axis=0).astype(float)
        winner = self._tiebreak(scores)
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'approval', 'policy': policy, 'approvals': approvals}
        )
    
    def score(
        self,
        utilities: np.ndarray,
        max_score: Optional[int] = None
    ) -> VotingResult:
        """
        Score voting: voters assign scores to candidates.
        
        Utilities are scaled to [0, max_score] range.
        
        Args:
            utilities: Shape (n_voters, n_candidates)
            max_score: Maximum score (defaults to config)
            
        Returns:
            VotingResult with total scores
        """
        max_score = max_score if max_score is not None else self.config.score_max
        
        # Scale utilities to [0, max_score]
        # Each voter's utilities scaled independently
        u_min = np.min(utilities, axis=1, keepdims=True)
        u_max = np.max(utilities, axis=1, keepdims=True)
        u_range = u_max - u_min
        u_range = np.where(u_range == 0, 1, u_range)  # Avoid division by zero
        
        normalized = (utilities - u_min) / u_range
        
        if self.config.score_granularity == 'integer':
            ballots = np.round(normalized * max_score).astype(int)
        else:
            ballots = normalized * max_score
        
        scores = np.sum(ballots, axis=0).astype(float)
        winner = self._tiebreak(scores)
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'score', 'max_score': max_score, 'ballots': ballots}
        )
    
    def star(
        self,
        utilities: np.ndarray,
        max_score: Optional[int] = None
    ) -> VotingResult:
        """
        STAR voting: Score Then Automatic Runoff.
        
        1. Sum scores to find top 2 candidates
        2. Head-to-head runoff between top 2
        
        Args:
            utilities: Shape (n_voters, n_candidates)
            max_score: Maximum score (defaults to config)
            
        Returns:
            VotingResult with STAR winner
        """
        max_score = max_score if max_score is not None else self.config.score_max
        
        # Phase 1: Score voting
        score_result = self.score(utilities, max_score)
        scores = score_result.scores
        
        # Find top 2 candidates
        top_2 = np.argsort(-scores)[:2]
        finalist_a, finalist_b = top_2[0], top_2[1]
        
        # Phase 2: Automatic runoff
        # Each voter's ballot contributes to whichever finalist they scored higher
        pref_a = np.sum(utilities[:, finalist_a] > utilities[:, finalist_b])
        pref_b = np.sum(utilities[:, finalist_b] > utilities[:, finalist_a])
        
        if pref_a > pref_b:
            winner = finalist_a
        elif pref_b > pref_a:
            winner = finalist_b
        else:
            # Tie: use higher score
            if scores[finalist_a] >= scores[finalist_b]:
                winner = finalist_a
            else:
                winner = finalist_b
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={
                'rule': 'star',
                'finalists': (finalist_a, finalist_b),
                'runoff_votes': (pref_a, pref_b)
            }
        )
    
    def median(self, utilities: np.ndarray) -> VotingResult:
        """
        Median voter rule: choose candidate with highest median utility.
        
        More robust to extreme preferences than utilitarian.
        
        Args:
            utilities: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        medians = np.median(utilities, axis=0)
        winner = self._tiebreak(medians)
        
        return VotingResult(
            winner=winner,
            scores=medians,
            metadata={'rule': 'median'}
        )
    
    def quadratic(self, utilities: np.ndarray) -> VotingResult:
        """
        Quadratic voting: votes weighted by square root.
        
        Reduces influence of extreme preferences.
        
        Args:
            utilities: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        # Apply square root to absolute utilities, preserve sign
        weighted = np.sign(utilities) * np.sqrt(np.abs(utilities))
        scores = np.sum(weighted, axis=0)
        winner = self._tiebreak(scores)
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'quadratic'}
        )
    
    # =========================================================================
    # Ordinal Rules (use rankings only)
    # =========================================================================
    
    def plurality(self, rankings: np.ndarray) -> VotingResult:
        """
        Plurality voting: most first-place votes wins.
        
        Args:
            rankings: Shape (n_voters, n_candidates) where
                     rankings[v, 0] is voter v's first choice
                     
        Returns:
            VotingResult with winner
        """
        n_voters, n_candidates = rankings.shape
        
        # Count first-place votes
        first_choices = rankings[:, 0]
        scores = np.bincount(first_choices, minlength=n_candidates).astype(float)
        winner = self._tiebreak(scores)
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'plurality'}
        )
    
    def anti_plurality(self, rankings: np.ndarray) -> VotingResult:
        """
        Anti-plurality (veto): candidate with fewest last-place votes wins.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        n_voters, n_candidates = rankings.shape
        
        # Count last-place votes (as negative score)
        last_choices = rankings[:, -1]
        last_counts = np.bincount(last_choices, minlength=n_candidates)
        
        # Score is negative of last-place count (fewer = better)
        scores = -last_counts.astype(float)
        winner = self._tiebreak(scores)
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'anti_plurality', 'last_place_counts': last_counts}
        )
    
    def veto(self, rankings: np.ndarray) -> VotingResult:
        """Alias for anti_plurality."""
        result = self.anti_plurality(rankings)
        result.metadata['rule'] = 'veto'
        return result
    
    def borda(self, rankings: np.ndarray) -> VotingResult:
        """
        Borda count: assign points based on rank position.
        
        First place gets n-1 points, second gets n-2, etc.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        n_voters, n_candidates = rankings.shape
        
        # Points: n-1 for first, n-2 for second, ..., 0 for last
        points = np.arange(n_candidates - 1, -1, -1)
        
        scores = np.zeros(n_candidates)
        for v in range(n_voters):
            for rank, candidate in enumerate(rankings[v]):
                scores[candidate] += points[rank]
        
        winner = self._tiebreak(scores)
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'borda'}
        )
    
    def irv(self, rankings: np.ndarray) -> VotingResult:
        """
        Instant Runoff Voting (IRV): eliminate candidates one by one.
        
        Repeatedly eliminate candidate with fewest first-place votes
        until one has majority.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        n_voters, n_candidates = rankings.shape
        
        # Track which candidates are still active
        active = np.ones(n_candidates, dtype=bool)
        rankings_copy = rankings.copy()
        
        elimination_order = []
        round_scores = []
        
        for round_num in range(n_candidates - 1):
            # Count first-place votes among active candidates
            first_choices = []
            for v in range(n_voters):
                for c in rankings_copy[v]:
                    if active[c]:
                        first_choices.append(c)
                        break
            
            vote_counts = np.bincount(first_choices, minlength=n_candidates).astype(float)
            vote_counts[~active] = -1  # Mark eliminated
            round_scores.append(vote_counts.copy())
            
            # Check for majority
            active_total = len(first_choices)
            if np.max(vote_counts) > active_total / 2:
                winner = np.argmax(vote_counts)
                break
            
            # Eliminate candidate with fewest votes
            active_counts = np.where(active, vote_counts, np.inf)
            eliminated = np.argmin(active_counts)
            active[eliminated] = False
            elimination_order.append(eliminated)
        else:
            # Last candidate standing
            winner = np.argmax(active)
        
        # Final scores are last round's vote counts
        scores = round_scores[-1] if round_scores else vote_counts
        scores[~active] = 0
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={
                'rule': 'irv',
                'elimination_order': elimination_order,
                'round_scores': round_scores
            }
        )
    
    def coombs(self, rankings: np.ndarray) -> VotingResult:
        """
        Coombs method: eliminate candidate with most last-place votes.
        
        Similar to IRV but eliminates by last-place votes.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        n_voters, n_candidates = rankings.shape
        
        active = np.ones(n_candidates, dtype=bool)
        elimination_order = []
        
        for round_num in range(n_candidates - 1):
            # Count first-place votes
            first_choices = []
            last_choices = []
            
            for v in range(n_voters):
                # Find first active
                for c in rankings[v]:
                    if active[c]:
                        first_choices.append(c)
                        break
                # Find last active
                for c in reversed(rankings[v]):
                    if active[c]:
                        last_choices.append(c)
                        break
            
            vote_counts = np.bincount(first_choices, minlength=n_candidates)
            last_counts = np.bincount(last_choices, minlength=n_candidates)
            
            # Check for majority
            active_total = len(first_choices)
            if np.max(vote_counts) > active_total / 2:
                winner = np.argmax(vote_counts)
                break
            
            # Eliminate candidate with most last-place votes
            active_last = np.where(active, last_counts, -np.inf)
            eliminated = np.argmax(active_last)
            active[eliminated] = False
            elimination_order.append(eliminated)
        else:
            winner = np.argmax(active)
        
        scores = np.zeros(n_candidates)
        scores[winner] = 1
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'coombs', 'elimination_order': elimination_order}
        )
    
    def condorcet(self, rankings: np.ndarray) -> VotingResult:
        """
        Condorcet method: find candidate who beats all others head-to-head.
        
        Returns winner or -1 if no Condorcet winner exists.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner (-1 if none)
        """
        margins = self._compute_pairwise_margins(rankings)
        n_candidates = margins.shape[0]
        
        # Condorcet winner beats everyone
        for c in range(n_candidates):
            if np.all(margins[c, np.arange(n_candidates) != c] > 0):
                scores = margins.sum(axis=1).astype(float)
                return VotingResult(
                    winner=c,
                    scores=scores,
                    metadata={'rule': 'condorcet', 'margins': margins}
                )
        
        # No Condorcet winner
        scores = margins.sum(axis=1).astype(float)
        return VotingResult(
            winner=-1,
            scores=scores,
            metadata={'rule': 'condorcet', 'margins': margins, 'no_winner': True}
        )
    
    def minimax(self, rankings: np.ndarray) -> VotingResult:
        """
        Minimax method: minimize worst pairwise defeat.
        
        Condorcet-consistent method that handles cycles.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        margins = self._compute_pairwise_margins(rankings)
        n_candidates = margins.shape[0]
        
        # For each candidate, find worst defeat (max margin against them)
        worst_defeats = np.zeros(n_candidates)
        for c in range(n_candidates):
            defeats = -margins[c, :]  # Positive = others beat c
            defeats[c] = -np.inf  # Ignore self
            worst_defeats[c] = np.max(defeats)
        
        # Score is negative of worst defeat (smaller worst defeat = better)
        scores = -worst_defeats
        winner = self._tiebreak(scores)
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'minimax', 'margins': margins, 'worst_defeats': worst_defeats}
        )
    
    def copeland(self, rankings: np.ndarray) -> VotingResult:
        """
        Copeland method: count pairwise wins minus losses.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        margins = self._compute_pairwise_margins(rankings)
        n_candidates = margins.shape[0]
        
        # Count wins and losses
        wins = np.sum(margins > 0, axis=1)
        losses = np.sum(margins < 0, axis=1)
        
        scores = (wins - losses).astype(float)
        winner = self._tiebreak(scores)
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'copeland', 'margins': margins, 'wins': wins, 'losses': losses}
        )
    
    def schulze(self, rankings: np.ndarray) -> VotingResult:
        """
        Schulze method (beatpath): find strongest paths.
        
        Condorcet-consistent method using transitive closure.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        margins = self._compute_pairwise_margins(rankings)
        n_candidates = margins.shape[0]
        
        # Compute strongest paths (Floyd-Warshall variant)
        strength = np.zeros((n_candidates, n_candidates))
        
        # Initialize with direct margins (only positive)
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j and margins[i, j] > 0:
                    strength[i, j] = margins[i, j]
        
        # Find strongest paths
        for k in range(n_candidates):
            for i in range(n_candidates):
                for j in range(n_candidates):
                    if i != j:
                        path_through_k = min(strength[i, k], strength[k, j])
                        if path_through_k > strength[i, j]:
                            strength[i, j] = path_through_k
        
        # Winner is candidate who has stronger path to all others
        scores = np.zeros(n_candidates)
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j:
                    if strength[i, j] > strength[j, i]:
                        scores[i] += 1
        
        winner = self._tiebreak(scores)
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'schulze', 'margins': margins, 'strength': strength}
        )
    
    def ranked_pairs(self, rankings: np.ndarray) -> VotingResult:
        """
        Ranked pairs (Tideman): lock pairs by margin strength.
        
        Condorcet-consistent method avoiding cycles.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        margins = self._compute_pairwise_margins(rankings)
        n_candidates = margins.shape[0]
        
        # Create list of pairs with positive margins
        pairs = []
        for i in range(n_candidates):
            for j in range(i + 1, n_candidates):
                if margins[i, j] > 0:
                    pairs.append((margins[i, j], i, j))
                elif margins[j, i] > 0:
                    pairs.append((margins[j, i], j, i))
        
        # Sort by margin strength (descending)
        pairs.sort(reverse=True)
        
        # Lock pairs, avoiding cycles
        locked = set()
        
        def creates_cycle(winner: int, loser: int) -> bool:
            """Check if adding winner > loser creates a cycle."""
            # BFS from loser to see if we can reach winner
            visited = {loser}
            queue = [loser]
            
            while queue:
                current = queue.pop(0)
                for (w, l) in locked:
                    if w == current and l not in visited:
                        if l == winner:
                            return True
                        visited.add(l)
                        queue.append(l)
            return False
        
        for margin, winner, loser in pairs:
            if not creates_cycle(winner, loser):
                locked.add((winner, loser))
        
        # Winner is source node (no one beats them in locked pairs)
        losers = {l for (w, l) in locked}
        for c in range(n_candidates):
            if c not in losers:
                scores = np.zeros(n_candidates)
                scores[c] = 1
                return VotingResult(
                    winner=c,
                    scores=scores,
                    metadata={'rule': 'ranked_pairs', 'margins': margins, 'locked': locked}
                )
        
        # Fallback (shouldn't happen)
        winner = 0
        scores = np.zeros(n_candidates)
        scores[winner] = 1
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'ranked_pairs', 'margins': margins, 'locked': locked}
        )
    
    def bucklin(self, rankings: np.ndarray) -> VotingResult:
        """
        Bucklin voting: expand rounds until majority.
        
        First, count first-place votes. If no majority, add second-place, etc.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        n_voters, n_candidates = rankings.shape
        majority = n_voters / 2
        
        scores = np.zeros(n_candidates)
        
        for round_num in range(n_candidates):
            # Add votes from this round
            for v in range(n_voters):
                candidate = rankings[v, round_num]
                scores[candidate] += 1
            
            # Check for majority
            if np.max(scores) > majority:
                winner = np.argmax(scores)
                return VotingResult(
                    winner=winner,
                    scores=scores,
                    metadata={'rule': 'bucklin', 'rounds_needed': round_num + 1}
                )
        
        # No majority ever (use highest count)
        winner = self._tiebreak(scores)
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'bucklin', 'rounds_needed': n_candidates}
        )
    
    def nanson(self, rankings: np.ndarray) -> VotingResult:
        """
        Nanson's method: iteratively eliminate below-average Borda candidates.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        n_voters, n_candidates = rankings.shape
        active = np.ones(n_candidates, dtype=bool)
        
        while np.sum(active) > 1:
            # Compute Borda scores for active candidates
            active_count = np.sum(active)
            points = np.arange(active_count - 1, -1, -1)
            
            scores = np.zeros(n_candidates)
            for v in range(n_voters):
                rank = 0
                for c in rankings[v]:
                    if active[c]:
                        scores[c] += points[rank]
                        rank += 1
            
            # Eliminate candidates with below-average score
            active_scores = scores[active]
            avg_score = np.mean(active_scores)
            
            for c in range(n_candidates):
                if active[c] and scores[c] < avg_score:
                    active[c] = False
            
            # If all remaining have same score, stop
            if np.sum(active) == active_count:
                break
        
        winner = np.argmax(active)
        scores = np.zeros(n_candidates)
        scores[winner] = 1
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'nanson'}
        )
    
    def baldwin(self, rankings: np.ndarray) -> VotingResult:
        """
        Baldwin's method: iteratively eliminate lowest Borda candidate.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        n_voters, n_candidates = rankings.shape
        active = np.ones(n_candidates, dtype=bool)
        elimination_order = []
        
        while np.sum(active) > 1:
            # Compute Borda scores for active candidates
            active_count = np.sum(active)
            points = np.arange(active_count - 1, -1, -1)
            
            scores = np.zeros(n_candidates)
            for v in range(n_voters):
                rank = 0
                for c in rankings[v]:
                    if active[c]:
                        scores[c] += points[rank]
                        rank += 1
            
            # Eliminate candidate with lowest score
            active_scores = np.where(active, scores, np.inf)
            eliminated = np.argmin(active_scores)
            active[eliminated] = False
            elimination_order.append(eliminated)
        
        winner = np.argmax(active)
        scores = np.zeros(n_candidates)
        scores[winner] = 1
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'baldwin', 'elimination_order': elimination_order}
        )
    
    def kemeny_young(self, rankings: np.ndarray) -> VotingResult:
        """
        Kemeny-Young method: find ranking that minimizes disagreement.
        
        Note: This is NP-hard; for small candidate sets only.
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            VotingResult with winner
        """
        from itertools import permutations
        
        n_voters, n_candidates = rankings.shape
        
        if n_candidates > 8:
            # Fall back to Schulze for large candidate sets
            result = self.schulze(rankings)
            result.metadata['rule'] = 'kemeny_young_approx'
            return result
        
        margins = self._compute_pairwise_margins(rankings)
        
        # Try all possible orderings
        best_score = -np.inf
        best_ordering = None
        
        for ordering in permutations(range(n_candidates)):
            # Compute Kemeny score for this ordering
            score = 0
            for i, c1 in enumerate(ordering):
                for c2 in ordering[i+1:]:
                    score += margins[c1, c2]
            
            if score > best_score:
                best_score = score
                best_ordering = ordering
        
        winner = best_ordering[0]
        scores = np.zeros(n_candidates)
        for i, c in enumerate(best_ordering):
            scores[c] = n_candidates - i
        
        return VotingResult(
            winner=winner,
            scores=scores,
            metadata={'rule': 'kemeny_young', 'optimal_ranking': best_ordering}
        )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _compute_pairwise_margins(self, rankings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise margin matrix.
        
        margins[i, j] = (voters preferring i to j) - (voters preferring j to i)
        
        Args:
            rankings: Shape (n_voters, n_candidates)
            
        Returns:
            Margin matrix (n_candidates, n_candidates)
        """
        n_voters, n_candidates = rankings.shape
        margins = np.zeros((n_candidates, n_candidates))
        
        for v in range(n_voters):
            for i in range(n_candidates):
                for j in range(i + 1, n_candidates):
                    c1, c2 = rankings[v, i], rankings[v, j]
                    # c1 is ranked higher (earlier) than c2
                    margins[c1, c2] += 1
                    margins[c2, c1] -= 1
        
        return margins
    
    def _tiebreak(self, scores: np.ndarray) -> int:
        """
        Break ties using configured method.
        
        Args:
            scores: Score array
            
        Returns:
            Winner index
        """
        max_score = np.max(scores)
        tied = np.where(scores == max_score)[0]
        
        if len(tied) == 1:
            return tied[0]
        
        method = self.config.tiebreak_method
        
        if method == 'random':
            return self.rng.choice(tied)
        elif method == 'lexicographic':
            return tied[0]
        else:  # 'none'
            return tied[0]
    
    def apply_rule(
        self,
        rule_name: str,
        utilities: Optional[np.ndarray] = None,
        rankings: Optional[np.ndarray] = None
    ) -> VotingResult:
        """
        Apply a voting rule by name.
        
        Args:
            rule_name: Name of the voting rule
            utilities: Utility matrix (for cardinal rules)
            rankings: Rankings matrix (for ordinal rules)
            
        Returns:
            VotingResult
        """
        cardinal_rules = {
            'utilitarian': self.utilitarian,
            'approval': self.approval,
            'score': self.score,
            'star': self.star,
            'median': self.median,
            'quadratic': self.quadratic,
        }
        
        ordinal_rules = {
            'plurality': self.plurality,
            'anti_plurality': self.anti_plurality,
            'veto': self.veto,
            'borda': self.borda,
            'irv': self.irv,
            'coombs': self.coombs,
            'condorcet': self.condorcet,
            'minimax': self.minimax,
            'copeland': self.copeland,
            'schulze': self.schulze,
            'ranked_pairs': self.ranked_pairs,
            'bucklin': self.bucklin,
            'nanson': self.nanson,
            'baldwin': self.baldwin,
            'kemeny_young': self.kemeny_young,
        }
        
        if rule_name in cardinal_rules:
            if utilities is None:
                raise ValueError(f"Cardinal rule '{rule_name}' requires utilities")
            return cardinal_rules[rule_name](utilities)
        
        elif rule_name in ordinal_rules:
            if rankings is None:
                raise ValueError(f"Ordinal rule '{rule_name}' requires rankings")
            return ordinal_rules[rule_name](rankings)
        
        else:
            raise ValueError(f"Unknown voting rule: {rule_name}")


# Convenience functions

def get_rule_type(rule_name: str) -> RuleType:
    """Get the type (cardinal/ordinal) of a voting rule."""
    cardinal = {'utilitarian', 'approval', 'score', 'star', 'median', 'quadratic'}
    ordinal = {
        'plurality', 'anti_plurality', 'veto', 'borda', 'irv', 'coombs',
        'condorcet', 'minimax', 'copeland', 'schulze', 'ranked_pairs',
        'bucklin', 'nanson', 'baldwin', 'kemeny_young'
    }
    
    if rule_name in cardinal:
        return RuleType.CARDINAL
    elif rule_name in ordinal:
        return RuleType.ORDINAL
    else:
        raise ValueError(f"Unknown voting rule: {rule_name}")


def list_voting_rules() -> Dict[str, Dict[str, str]]:
    """Get dictionary of all voting rules with descriptions."""
    from .config import AVAILABLE_VOTING_RULES
    return AVAILABLE_VOTING_RULES




