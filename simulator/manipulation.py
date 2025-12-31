"""
Strategic manipulation module for voting simulation.

Implements various manipulation strategies and voter selection methods
for studying strategic voting behavior.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from .config import ManipulationConfig
from .voting_rules import VotingRuleEngine, VotingResult, get_rule_type, RuleType


@dataclass
class ManipulationResult:
    """Result of applying manipulation to an election."""
    
    # Original sincere ballots
    sincere_utilities: np.ndarray
    sincere_rankings: np.ndarray
    
    # Manipulated ballots
    manipulated_utilities: np.ndarray
    manipulated_rankings: np.ndarray
    
    # Which voters are manipulators
    manipulator_mask: np.ndarray  # Boolean array
    
    # Information about manipulation
    strategy: str
    n_manipulators: int
    manipulation_details: Dict[str, Any]


class ManipulationEngine:
    """
    Engine for applying strategic manipulation to elections.
    
    Supports various manipulation strategies and voter selection methods.
    """
    
    def __init__(
        self,
        config: ManipulationConfig,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize manipulation engine.
        
        Args:
            config: Manipulation configuration
            rng: Random generator for reproducibility
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def select_manipulators(
        self,
        utilities: np.ndarray,
        n_voters: int
    ) -> np.ndarray:
        """
        Select which voters will manipulate.
        
        Args:
            utilities: Utility matrix (n_voters, n_candidates)
            n_voters: Number of voters
            
        Returns:
            Boolean mask of manipulators
        """
        n_manipulators = int(self.config.manipulator_fraction * n_voters)
        n_manipulators = max(0, min(n_manipulators, n_voters))
        
        method = self.config.selection_method
        
        if method == 'random':
            return self._select_random(n_voters, n_manipulators)
        
        elif method == 'extremists':
            return self._select_extremists(utilities, n_manipulators)
        
        elif method == 'centrists':
            return self._select_centrists(utilities, n_manipulators)
        
        elif method == 'informed':
            return self._select_informed(utilities, n_manipulators)
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _select_random(self, n_voters: int, n_manipulators: int) -> np.ndarray:
        """Randomly select manipulators."""
        mask = np.zeros(n_voters, dtype=bool)
        indices = self.rng.choice(n_voters, n_manipulators, replace=False)
        mask[indices] = True
        return mask
    
    def _select_extremists(
        self,
        utilities: np.ndarray,
        n_manipulators: int
    ) -> np.ndarray:
        """
        Select voters with most extreme preferences.
        
        Extremism measured by variance of utilities.
        """
        n_voters = utilities.shape[0]
        variance = np.var(utilities, axis=1)
        
        # Most extreme = highest variance
        indices = np.argsort(-variance)[:n_manipulators]
        mask = np.zeros(n_voters, dtype=bool)
        mask[indices] = True
        return mask
    
    def _select_centrists(
        self,
        utilities: np.ndarray,
        n_manipulators: int
    ) -> np.ndarray:
        """
        Select voters with most centrist preferences.
        
        Centrist = utilities closer to average.
        """
        n_voters = utilities.shape[0]
        
        # Distance from average utility pattern
        avg_utility = np.mean(utilities, axis=0)
        distances = np.sum((utilities - avg_utility) ** 2, axis=1)
        
        # Most centrist = smallest distance
        indices = np.argsort(distances)[:n_manipulators]
        mask = np.zeros(n_voters, dtype=bool)
        mask[indices] = True
        return mask
    
    def _select_informed(
        self,
        utilities: np.ndarray,
        n_manipulators: int
    ) -> np.ndarray:
        """
        Select voters who have most to gain from manipulation.
        
        Based on difference between best and expected outcome.
        """
        n_voters, n_candidates = utilities.shape
        
        # Approximate expected winner (simple plurality)
        rankings = np.argsort(-utilities, axis=1)
        first_choices = rankings[:, 0]
        vote_counts = np.bincount(first_choices, minlength=n_candidates)
        expected_winner = np.argmax(vote_counts)
        
        # Utility gain potential = utility(best) - utility(expected winner)
        best_utilities = np.max(utilities, axis=1)
        expected_utilities = utilities[:, expected_winner]
        gain_potential = best_utilities - expected_utilities
        
        # Select those with most to gain
        indices = np.argsort(-gain_potential)[:n_manipulators]
        mask = np.zeros(n_voters, dtype=bool)
        mask[indices] = True
        return mask
    
    def get_poll_information(
        self,
        utilities: np.ndarray,
        rule_name: str
    ) -> Dict[str, Any]:
        """
        Generate poll information for manipulators.
        
        Args:
            utilities: Utility matrix
            rule_name: Voting rule being used
            
        Returns:
            Poll information dictionary
        """
        n_voters, n_candidates = utilities.shape
        rankings = np.argsort(-utilities, axis=1)
        
        # Basic plurality poll
        first_choices = rankings[:, 0]
        vote_counts = np.bincount(first_choices, minlength=n_candidates)
        
        # Add noise
        noise = self.config.poll_noise
        noisy_counts = vote_counts + self.rng.normal(0, noise * n_voters, n_candidates)
        noisy_counts = np.maximum(0, noisy_counts)
        noisy_shares = noisy_counts / np.sum(noisy_counts)
        
        return {
            'vote_shares': noisy_shares,
            'expected_top_2': np.argsort(-noisy_shares)[:2],
            'frontrunner': np.argmax(noisy_shares),
            'viable_candidates': np.where(noisy_shares > 0.1)[0],
        }
    
    def apply_manipulation(
        self,
        utilities: np.ndarray,
        rankings: np.ndarray,
        rule_name: str
    ) -> ManipulationResult:
        """
        Apply manipulation strategy to an election.
        
        Args:
            utilities: Sincere utility matrix (n_voters, n_candidates)
            rankings: Sincere rankings (n_voters, n_candidates)
            rule_name: Voting rule being used
            
        Returns:
            ManipulationResult with manipulated ballots
        """
        if not self.config.enabled:
            return ManipulationResult(
                sincere_utilities=utilities,
                sincere_rankings=rankings,
                manipulated_utilities=utilities.copy(),
                manipulated_rankings=rankings.copy(),
                manipulator_mask=np.zeros(utilities.shape[0], dtype=bool),
                strategy='none',
                n_manipulators=0,
                manipulation_details={}
            )
        
        n_voters, n_candidates = utilities.shape
        
        # Select manipulators
        manipulator_mask = self.select_manipulators(utilities, n_voters)
        n_manipulators = np.sum(manipulator_mask)
        
        # Get poll information
        poll_info = None
        if self.config.information_level == 'polls':
            poll_info = self.get_poll_information(utilities, rule_name)
        elif self.config.information_level == 'full':
            # Full information: exact vote shares
            first_choices = rankings[:, 0]
            vote_counts = np.bincount(first_choices, minlength=n_candidates)
            poll_info = {
                'vote_shares': vote_counts / n_voters,
                'expected_top_2': np.argsort(-vote_counts)[:2],
                'frontrunner': np.argmax(vote_counts),
                'viable_candidates': np.where(vote_counts > n_voters * 0.1)[0],
            }
        
        # Apply manipulation strategy
        strategy = self.config.strategy
        
        manipulated_utilities = utilities.copy()
        manipulated_rankings = rankings.copy()
        
        if strategy == 'bullet':
            self._apply_bullet(
                manipulated_utilities, manipulated_rankings,
                utilities, rankings, manipulator_mask, rule_name
            )
        
        elif strategy == 'compromise':
            self._apply_compromise(
                manipulated_utilities, manipulated_rankings,
                utilities, rankings, manipulator_mask, poll_info
            )
        
        elif strategy == 'burial':
            self._apply_burial(
                manipulated_utilities, manipulated_rankings,
                utilities, rankings, manipulator_mask, poll_info
            )
        
        elif strategy == 'pushover':
            self._apply_pushover(
                manipulated_utilities, manipulated_rankings,
                utilities, rankings, manipulator_mask, poll_info
            )
        
        elif strategy == 'optimal':
            self._apply_optimal(
                manipulated_utilities, manipulated_rankings,
                utilities, rankings, manipulator_mask, rule_name
            )
        
        else:
            raise ValueError(f"Unknown manipulation strategy: {strategy}")
        
        return ManipulationResult(
            sincere_utilities=utilities,
            sincere_rankings=rankings,
            manipulated_utilities=manipulated_utilities,
            manipulated_rankings=manipulated_rankings,
            manipulator_mask=manipulator_mask,
            strategy=strategy,
            n_manipulators=n_manipulators,
            manipulation_details={
                'poll_info': poll_info,
                'information_level': self.config.information_level,
            }
        )
    
    def _apply_bullet(
        self,
        manip_utilities: np.ndarray,
        manip_rankings: np.ndarray,
        sincere_utilities: np.ndarray,
        sincere_rankings: np.ndarray,
        mask: np.ndarray,
        rule_name: str
    ) -> None:
        """
        Bullet voting: only support top choice.
        
        For cardinal rules: give max to top choice, 0 to others.
        For ordinal rules: no change (not applicable).
        """
        rule_type = get_rule_type(rule_name)
        
        if rule_type == RuleType.CARDINAL:
            # Set utilities: max for first choice, 0 for others
            for v in np.where(mask)[0]:
                top_choice = sincere_rankings[v, 0]
                max_util = sincere_utilities[v, top_choice]
                manip_utilities[v, :] = 0
                manip_utilities[v, top_choice] = max_util
    
    def _apply_compromise(
        self,
        manip_utilities: np.ndarray,
        manip_rankings: np.ndarray,
        sincere_utilities: np.ndarray,
        sincere_rankings: np.ndarray,
        mask: np.ndarray,
        poll_info: Optional[Dict]
    ) -> None:
        """
        Compromise: raise ranking of viable alternative.
        
        If top choice isn't viable, boost second-choice viable candidate.
        """
        if poll_info is None:
            return
        
        viable = set(poll_info.get('viable_candidates', []))
        frontrunner = poll_info.get('frontrunner', 0)
        
        for v in np.where(mask)[0]:
            ranking = sincere_rankings[v].copy()
            top_choice = ranking[0]
            
            # If top choice isn't viable, find best viable and promote
            if top_choice not in viable and len(viable) > 0:
                # Find first viable in their ranking
                for rank, candidate in enumerate(ranking):
                    if candidate in viable:
                        # Move this candidate to first place
                        manip_rankings[v] = np.roll(ranking, -rank)
                        manip_rankings[v][0] = candidate
                        # Shift others down
                        others = [c for c in ranking if c != candidate]
                        manip_rankings[v][1:] = others[:len(ranking)-1]
                        break
    
    def _apply_burial(
        self,
        manip_utilities: np.ndarray,
        manip_rankings: np.ndarray,
        sincere_utilities: np.ndarray,
        sincere_rankings: np.ndarray,
        mask: np.ndarray,
        poll_info: Optional[Dict]
    ) -> None:
        """
        Burial: rank viable competitor lower.
        
        Move frontrunner to last place if they're not the voter's top choice.
        """
        if poll_info is None:
            return
        
        frontrunner = poll_info.get('frontrunner', 0)
        
        for v in np.where(mask)[0]:
            ranking = sincere_rankings[v].copy()
            top_choice = ranking[0]
            
            if frontrunner != top_choice:
                # Move frontrunner to last place
                ranking_list = ranking.tolist()
                if frontrunner in ranking_list:
                    ranking_list.remove(frontrunner)
                    ranking_list.append(frontrunner)
                    manip_rankings[v] = np.array(ranking_list)
    
    def _apply_pushover(
        self,
        manip_utilities: np.ndarray,
        manip_rankings: np.ndarray,
        sincere_utilities: np.ndarray,
        sincere_rankings: np.ndarray,
        mask: np.ndarray,
        poll_info: Optional[Dict]
    ) -> None:
        """
        Pushover: support weak opponent to face in runoff.
        
        Rank weakest candidate first to help them advance.
        """
        if poll_info is None:
            return
        
        vote_shares = poll_info.get('vote_shares', np.ones(sincere_rankings.shape[1]))
        weakest = np.argmin(vote_shares)
        
        for v in np.where(mask)[0]:
            ranking = sincere_rankings[v].copy()
            top_choice = ranking[0]
            
            # Only apply if weakest isn't already top choice
            if weakest != top_choice:
                ranking_list = ranking.tolist()
                ranking_list.remove(weakest)
                # Put weakest second (after true top choice)
                ranking_list.insert(1, weakest)
                manip_rankings[v] = np.array(ranking_list)
    
    def _apply_optimal(
        self,
        manip_utilities: np.ndarray,
        manip_rankings: np.ndarray,
        sincere_utilities: np.ndarray,
        sincere_rankings: np.ndarray,
        mask: np.ndarray,
        rule_name: str
    ) -> None:
        """
        Optimal manipulation: compute best strategic vote.
        
        Expensive - tries different strategies and picks best.
        Note: This is a simplified version for small candidate sets.
        """
        from itertools import permutations
        from .config import VotingRuleConfig
        
        n_voters, n_candidates = sincere_utilities.shape
        
        if n_candidates > 5:
            # Too expensive, fall back to compromise
            self._apply_compromise(
                manip_utilities, manip_rankings,
                sincere_utilities, sincere_rankings,
                mask, None
            )
            return
        
        engine = VotingRuleEngine(VotingRuleConfig(), self.rng)
        
        for v in np.where(mask)[0]:
            best_ranking = sincere_rankings[v].copy()
            best_utility = -np.inf
            
            # Try all possible rankings
            for perm in permutations(range(n_candidates)):
                test_rankings = manip_rankings.copy()
                test_rankings[v] = np.array(perm)
                
                # Compute winner with this ranking
                test_utilities = manip_utilities.copy()
                
                try:
                    rule_type = get_rule_type(rule_name)
                    if rule_type == RuleType.CARDINAL:
                        result = engine.apply_rule(rule_name, utilities=test_utilities)
                    else:
                        result = engine.apply_rule(rule_name, rankings=test_rankings)
                    
                    winner = result.winner
                    if winner >= 0:
                        utility = sincere_utilities[v, winner]
                        if utility > best_utility:
                            best_utility = utility
                            best_ranking = np.array(perm)
                except Exception:
                    continue
            
            manip_rankings[v] = best_ranking


def compute_manipulation_impact(
    sincere_result: VotingResult,
    manipulated_result: VotingResult,
    sincere_utilities: np.ndarray,
    manipulator_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Compute the impact of manipulation on election outcome.
    
    Args:
        sincere_result: Result with sincere voting
        manipulated_result: Result with manipulation
        sincere_utilities: True utilities of voters
        manipulator_mask: Boolean mask of manipulators
        
    Returns:
        Dictionary with impact metrics
    """
    sincere_winner = sincere_result.winner
    manip_winner = manipulated_result.winner
    
    # Did winner change?
    winner_changed = sincere_winner != manip_winner
    
    # Utility changes
    n_voters = sincere_utilities.shape[0]
    
    # Social utility change
    sincere_social_utility = np.mean(sincere_utilities[:, sincere_winner])
    manip_social_utility = np.mean(sincere_utilities[:, manip_winner])
    social_utility_change = manip_social_utility - sincere_social_utility
    
    # Manipulator utility change
    if np.any(manipulator_mask):
        sincere_manip_utility = np.mean(
            sincere_utilities[manipulator_mask, sincere_winner]
        )
        manip_manip_utility = np.mean(
            sincere_utilities[manipulator_mask, manip_winner]
        )
        manipulator_gain = manip_manip_utility - sincere_manip_utility
    else:
        manipulator_gain = 0.0
    
    # Non-manipulator utility change
    honest_mask = ~manipulator_mask
    if np.any(honest_mask):
        sincere_honest_utility = np.mean(
            sincere_utilities[honest_mask, sincere_winner]
        )
        manip_honest_utility = np.mean(
            sincere_utilities[honest_mask, manip_winner]
        )
        honest_loss = sincere_honest_utility - manip_honest_utility
    else:
        honest_loss = 0.0
    
    return {
        'winner_changed': winner_changed,
        'sincere_winner': sincere_winner,
        'manipulated_winner': manip_winner,
        'social_utility_change': social_utility_change,
        'manipulator_gain': manipulator_gain,
        'honest_voter_loss': honest_loss,
        'manipulation_successful': winner_changed and manipulator_gain > 0,
    }



