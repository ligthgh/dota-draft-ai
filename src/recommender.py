from __future__ import annotations

from dataclasses import dataclass

from .hero_roles import hero_can_play_role
from .nn_predictor import NeuralDraftPredictor


@dataclass
class Recommendation:
    hero_id: int
    win_probability: float
    delta_vs_baseline: float


class LastPickRecommender:
    """Side-neutral last-pick recommender.

    The user supplies only "my team" and "enemy team". To suppress Radiant/Dire
    bias in a model trained on Radiant-win labels, every completed draft is scored
    in both orientations and the two estimates are averaged.
    """

    def __init__(self, predictor: NeuralDraftPredictor):
        self.predictor = predictor

    def team_win_probability(self, my_five: list[int], enemy_five: list[int]) -> float:
        p_as_radiant = self.predictor.radiant_win_probability(my_five, enemy_five)
        p_enemy_as_radiant = self.predictor.radiant_win_probability(enemy_five, my_five)
        p_as_dire = 1.0 - p_enemy_as_radiant
        return (p_as_radiant + p_as_dire) / 2.0

    def recommend(
        self,
        my_four: list[int],
        enemy_five: list[int],
        role: int,
        top_k: int = 10,
    ) -> list[Recommendation]:
        if len(my_four) != 4:
            raise ValueError("Моя команда должна содержать ровно 4 выбранных героя.")
        if len(enemy_five) != 5:
            raise ValueError("Команда противника должна содержать ровно 5 героев.")
        if len(set(my_four + enemy_five)) != 9:
            raise ValueError("В драфте есть повторяющиеся герои.")

        occupied = set(my_four + enemy_five)
        scored = []

        for hero_id in self.predictor.indexer.hero_to_idx.keys():
            if hero_id in occupied:
                continue
            if not hero_can_play_role(hero_id, role):
                continue

            probability = self.team_win_probability(
                my_four + [hero_id],
                enemy_five,
            )
            scored.append((hero_id, probability))

        if not scored:
            return []

        baseline = sum(p for _, p in scored) / len(scored)
        result = [
            Recommendation(
                hero_id=hero_id,
                win_probability=probability,
                delta_vs_baseline=probability - baseline,
            )
            for hero_id, probability in scored
        ]
        result.sort(key=lambda x: x.win_probability, reverse=True)
        return result[:top_k]
