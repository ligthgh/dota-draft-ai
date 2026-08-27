from __future__ import annotations

from dataclasses import dataclass

from .hero_roles import hero_can_play_role
from .nn_predictor_v8 import NeuralDraftPredictorV8


@dataclass
class Recommendation:
    hero_id: int
    win_probability: float
    delta_vs_average: float


class LastPickRecommenderV8:
    def __init__(self, predictor: NeuralDraftPredictorV8):
        self.predictor = predictor

    @staticmethod
    def build_team_by_position(
        known_heroes: list[int],
        known_positions: list[int],
        missing_position: int,
        missing_hero: int = 0,
        known_picks: list[int] | None = None,
        missing_pick: int = 0,
    ):
        heroes = [0, 0, 0, 0, 0]
        picks = [0, 0, 0, 0, 0]
        positions = [1, 2, 3, 4, 5]

        known_picks = known_picks or [0] * len(known_heroes)

        for hero, pos, pick in zip(
            known_heroes,
            known_positions,
            known_picks,
        ):
            heroes[pos - 1] = int(hero)
            picks[pos - 1] = int(pick)

        heroes[missing_position - 1] = int(missing_hero)
        picks[missing_position - 1] = int(missing_pick)

        return heroes, positions, picks

    def recommend(
        self,
        my_four,
        enemy_four,
        my_positions,
        enemy_positions,
        needed_role,
        enemy_missing_role,
        my_picks,
        enemy_picks,
        candidate_pick_order,
        avg_mmr,
        rank_bracket,
        top_k=10,
    ):
        occupied = set(my_four + enemy_four)
        scored = []

        enemy_heroes5, enemy_pos5, enemy_picks5 = self.build_team_by_position(
            enemy_four,
            enemy_positions,
            enemy_missing_role,
            missing_hero=0,
            known_picks=enemy_picks,
            missing_pick=0,
        )

        for hero_id in self.predictor.indexer.hero_to_idx.keys():
            if hero_id in occupied:
                continue

            if not hero_can_play_role(
                hero_id,
                needed_role,
            ):
                continue

            my_heroes5, my_pos5, my_picks5 = self.build_team_by_position(
                my_four,
                my_positions,
                needed_role,
                missing_hero=hero_id,
                known_picks=my_picks,
                missing_pick=candidate_pick_order,
            )

            p_as_radiant = self.predictor.probability(
                my_heroes5,
                enemy_heroes5,
                my_pos5,
                enemy_pos5,
                my_picks5,
                enemy_picks5,
                avg_mmr,
                rank_bracket,
            )

            p_enemy_as_radiant = self.predictor.probability(
                enemy_heroes5,
                my_heroes5,
                enemy_pos5,
                my_pos5,
                enemy_picks5,
                my_picks5,
                avg_mmr,
                rank_bracket,
            )

            p = (
                p_as_radiant
                + (1.0 - p_enemy_as_radiant)
            ) / 2.0

            scored.append(
                (hero_id, p)
            )

        if not scored:
            return []

        average = sum(
            p for _, p in scored
        ) / len(scored)

        result = [
            Recommendation(
                hero_id=hero_id,
                win_probability=p,
                delta_vs_average=p - average,
            )
            for hero_id, p in scored
        ]

        result.sort(
            key=lambda x: x.win_probability,
            reverse=True,
        )

        return result[:top_k]
