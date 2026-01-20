import dataclasses
import functools
import itertools
from functools import lru_cache
from typing import Generator
from enum import Enum, auto
from unittest import case

import mole_game


def flatten(*xss) -> list:
    return [x for xs in xss for x in (flatten(*xs) if isinstance(xs, (list, tuple, set)) else (xs,))]


@dataclasses.dataclass(frozen=True)
class SolverField:
    hidden: bool = True
    is_mole: bool | None = None
    neighbour_moles: int | None = None


@dataclasses.dataclass(frozen=True)
class SolverPosition:
    size: int
    board: tuple[tuple[SolverField, ...], ...]
    remaining_targets: int


@dataclasses.dataclass(frozen=True)
class Coordinate:
    row: int
    col: int

    def __getitem__(self, key: int) -> int:
        match key:
            case 0:
                return self.row
            case 1:
                return self.col
            case _:
                raise IndexError

    def __str__(self):
        """I accidentally flipped how I draw the board in the GUI, so instead of (row, col), this now specifies (col, row),
        which might actually be more intuitive to read as (x, y)"""
        return f"(r={self.row + 1}, c={self.col + 1})"


class Solver:
    INF: int = 10 ** 9

    position: SolverPosition
    best_move: Coordinate | None
    most_shots_necessary: int

    def __init__(self, size: int, number_of_targets: int):
        self.position = SolverPosition(size, tuple(), number_of_targets)
        self.best_move = None
        self.most_shots_necessary = self.position.size ** 2

    def _reset_board(self):
        board = tuple(tuple(SolverField() for _ in range(self.position.size)) for _ in range(self.position.size))
        self.position = SolverPosition(size=self.position.size, board=board,
                                       remaining_targets=self.position.remaining_targets)

    def load_position(self, game: mole_game.Game):
        self._reset_board()

        revealed_moles: int = sum(
            1 for row in range(self.position.size)
            for col in range(self.position.size)
            if game.board[row][col].is_revealed and game.board[row][col].has_mole
        )

        board: tuple[tuple[SolverField, ...], ...] = tuple(tuple(SolverField(
            hidden=(not (revealed := game.board[row][col].is_revealed)),
            is_mole=(is_mole := game.board[row][col].has_mole) if revealed else None,
            neighbour_moles=game.board[row][col].neighbours if revealed and not is_mole else None,
        ) for col in range(self.position.size)) for row in range(self.position.size))
        self.position = SolverPosition(size=self.position.size, board=board,
                                       remaining_targets=self.position.remaining_targets - revealed_moles)

        self.best_move = None
        self.most_shots_necessary = sum(1 for field in flatten(self.position.board) if field.hidden)

    @staticmethod
    def neighbours(size: int, row: int, col: int) -> Generator[Coordinate]:
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue

                neighbour_row: int = row + delta_row
                neighbour_col: int = col + delta_col

                if 0 <= neighbour_row < size and 0 <= neighbour_col < size:
                    yield Coordinate(neighbour_row, neighbour_col)

    @staticmethod
    def get_all_cells(position: SolverPosition) -> Generator[tuple[SolverField, int, int]]:
        for row in range(position.size):
            for col in range(position.size):
                yield position.board[row][col], row, col

    @staticmethod
    def get_frontier(position: SolverPosition) -> set[Coordinate]:
        """Returns coordinates of all unrevealed fields that neighbour a numbered cell"""
        frontier: set[Coordinate] = set()

        for cell, row, col in Solver.get_all_cells(position):
            if cell.hidden or cell.is_mole:
                continue

            for neighbour_coordinate in Solver.neighbours(position.size, row, col):
                neighbour_row, neighbour_col = neighbour_coordinate
                if position.board[neighbour_row][neighbour_col].is_mole is None:
                    frontier.add(neighbour_coordinate)

        return frontier

    @staticmethod
    def get_constraints(position: SolverPosition, frontiers: set[Coordinate]) -> list[tuple[
        list[Coordinate],
        int
    ]]:
        constraints: list[tuple[
            list[Coordinate],
            int
        ]] = []

        for cell, row, col in Solver.get_all_cells(position):
            if cell.hidden or cell.is_mole:
                continue

            unknown_frontier_neighbours: list[Coordinate] = []
            revealed_surrounding_moles: int = 0
            for neighbour_coordinate in Solver.neighbours(position.size, row, col):
                neighbour_row, neighbour_col = neighbour_coordinate
                neighbour_cell: SolverField = position.board[neighbour_row][neighbour_col]
                if neighbour_cell.is_mole is True:
                    revealed_surrounding_moles += 1
                elif neighbour_cell.is_mole is None and neighbour_coordinate in frontiers:
                    unknown_frontier_neighbours.append(neighbour_coordinate)

            required: int = cell.neighbour_moles - revealed_surrounding_moles
            constraints.append((unknown_frontier_neighbours, required))

        return constraints

    @staticmethod
    def constraints_still_possible(assignment: dict[Coordinate, bool], constraints: list[tuple[
        list[Coordinate],
        int
    ]]):
        for cells, required_moles in constraints:
            assigned = [assignment[cell] for cell in cells if cell in assignment]
            assigned_moles: int = sum(assigned)
            unassigned_cells: int = len(cells) - len(assigned)

            if assigned_moles > required_moles:
                return False

            # not enough empty space to reach required moles
            if assigned_moles + unassigned_cells < required_moles:
                return False

        return True

    @staticmethod
    @lru_cache(maxsize=None)
    def cached_generate_frontier_assignments(position: SolverPosition) -> tuple[dict[Coordinate, bool], ...]:
        return tuple(Solver.generate_frontier_assignments(position))

    @staticmethod
    def generate_frontier_assignments(position: SolverPosition) -> Generator[dict[Coordinate, bool]]:
        frontier_set: set[Coordinate] = Solver.get_frontier(position)
        frontier: list[Coordinate] = list(frontier_set)
        constraints = Solver.get_constraints(position, frontier_set)
        known_moles: int = sum(1 for cell, _, _ in Solver.get_all_cells(position) if cell.is_mole is True)
        remaining_moles: int = position.remaining_targets - known_moles
        deep_unknown_count: int = sum(1 for cell, row, col in Solver.get_all_cells(position)
                                      if cell.is_mole is None
                                      and cell.hidden
                                      and Coordinate(row, col) not in frontier_set)

        assignment: dict[Coordinate, bool] = {}

        def backtrack(frontier_idx: int, moles_left: int):
            if frontier_idx == len(frontier):
                # remaining moles must fit into deep unknowns
                if 0 <= moles_left <= deep_unknown_count:
                    yield assignment.copy()

                return None

            cell: Coordinate = frontier[frontier_idx]

            # try no mole
            assignment[cell] = False
            if Solver.constraints_still_possible(assignment, constraints):
                yield from backtrack(frontier_idx + 1, moles_left)

            # try mole
            if moles_left > 0:
                assignment[cell] = True
                if Solver.constraints_still_possible(assignment, constraints):
                    yield from backtrack(frontier_idx + 1, moles_left - 1)

            del assignment[cell]

        yield from backtrack(0, remaining_moles)

    @staticmethod
    def apply_move(position: SolverPosition, move: Coordinate, outcome_is_mole: bool,
                   neighbour_count: int) -> SolverPosition:
        board: tuple[tuple[SolverField, ...], ...] = tuple(
            tuple(
                SolverField(
                    hidden=(False if (row == move.row and col == move.col) else cell.hidden),
                    is_mole=(outcome_is_mole if (row == move.row and col == move.col) else cell.is_mole),
                    neighbour_moles=(neighbour_count if (
                            row == move.row and col == move.col and not outcome_is_mole) else cell.neighbour_moles)
                )
                for col, cell in enumerate(row_data)
            )
            for row, row_data in enumerate(position.board)
        )

        return SolverPosition(
            size=position.size,
            board=board,
            remaining_targets=position.remaining_targets - outcome_is_mole,
        )

    @staticmethod
    def outcomes_for_move(position: SolverPosition, move: Coordinate) -> Generator[tuple[bool, int | None]]:
        frontier: set[Coordinate] = Solver.get_frontier(position)

        if move not in frontier:
            # deep unknown cell => unconstrained
            yield True, None
            yield False, 0
            return

        # frontier cell:
        seen: set[tuple[bool, int]] = set()
        for assignment in Solver.cached_generate_frontier_assignments(position):
            if move not in assignment:
                continue

            is_mole: bool = assignment[move]
            if is_mole:
                outcome = is_mole, None
            else:
                neighbour_moles: int = sum(1
                                           for neighbour in Solver.neighbours(position.size, move.row, move.col)
                                           if assignment.get(neighbour) is True)
                outcome = is_mole, neighbour_moles

            if outcome not in seen:
                seen.add(outcome)
                yield outcome



    @staticmethod
    def position_is_consistent(position: SolverPosition) -> bool:
        hidden_cells: int = 0
        for cell, row, col in Solver.get_all_cells(position):
            hidden_cells += cell.hidden

            if cell.hidden or cell.is_mole:
                continue

            known_neighbouring_moles: int = 0
            unknown_neighbouring_moles: int = 0

            for neighbour_row, neighbour_col in Solver.neighbours(position.size, row, col):
                neighbour: SolverField = position.board[neighbour_row][neighbour_col]
                if neighbour.is_mole is True:
                    known_neighbouring_moles += 1
                elif neighbour.is_mole is None:
                    unknown_neighbouring_moles += 1

            if known_neighbouring_moles > cell.neighbour_moles:
                return False

            if known_neighbouring_moles + unknown_neighbouring_moles < cell.neighbour_moles:
                return False

        if not 0 <= position.remaining_targets <= hidden_cells:
            return False

        return True

    @staticmethod
    @lru_cache(maxsize=None)
    def solve(position: SolverPosition) -> int:
        if position.remaining_targets == 0:
            return 0

        unrevealed_coordinates: list[Coordinate] = [Coordinate(row, col) for cell, row, col in
                                                    Solver.get_all_cells(position) if cell.hidden]

        if not unrevealed_coordinates:
            return Solver.INF  # this should never happen

        best_score: int = Solver.INF

        for move in unrevealed_coordinates:
            worst_score: int = 0

            for is_mole, neighbour_moles in Solver.outcomes_for_move(position, move):
                next_pos: SolverPosition = Solver.apply_move(position, move, is_mole, neighbour_moles)
                if not Solver.position_is_consistent(next_pos):
                    continue  # impossible position

                worst_score = max(worst_score, Solver.solve(next_pos))

                # pruning
                if worst_score >= best_score:
                    break

            best_score = min(best_score, worst_score)

        return best_score

    def calculate_best_field(self):
        best_score: int = Solver.INF
        best_move: Coordinate | None = None

        for cell, row, col in Solver.get_all_cells(self.position):
            if not cell.hidden:
                continue

            move: Coordinate = Coordinate(row, col)
            worst_score: int = 0

            for is_mole, neighbour_moles in Solver.outcomes_for_move(self.position, move):
                next_pos: SolverPosition = Solver.apply_move(self.position, move, is_mole, neighbour_moles)
                if not Solver.position_is_consistent(next_pos):
                    continue

                worst_score = max(worst_score, Solver.solve(next_pos))

                # pruning
                if worst_score >= best_score:
                    break

            if worst_score < best_score:
                best_score = worst_score
                best_move = move

        self.best_move = best_move
        self.most_shots_necessary = best_score

    def stop_calculation(self):
        pass
