import dataclasses
from functools import lru_cache
from typing import Iterator

from mole_game import Game


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
    INF: int = 1000

    number_of_targets: int
    position: SolverPosition
    best_move: Coordinate | None
    most_shots_necessary: int

    def __init__(self, size: int, number_of_targets: int):
        self.number_of_targets = number_of_targets
        self.position = SolverPosition(size, tuple(), self.number_of_targets)
        self.best_move = None
        self.most_shots_necessary = self.position.size ** 2

    def _reset_board(self):
        board = tuple(tuple(SolverField() for _ in range(self.position.size)) for _ in range(self.position.size))
        self.position = SolverPosition(size=self.position.size, board=board,
                                       remaining_targets=self.number_of_targets)

    def load_position(self, game: Game):
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
                                       remaining_targets=self.number_of_targets - revealed_moles)

        self.best_move = None
        self.most_shots_necessary = sum(1 for field in flatten(self.position.board) if field.hidden)

    @staticmethod
    def neighbours(size: int, row: int, col: int) -> Iterator[Coordinate]:
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue

                neighbour_row: int = row + delta_row
                neighbour_col: int = col + delta_col

                if 0 <= neighbour_row < size and 0 <= neighbour_col < size:
                    yield Coordinate(neighbour_row, neighbour_col)

    @staticmethod
    def get_all_cells(position: SolverPosition) -> Iterator[tuple[SolverField, int, int]]:
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
    def generate_frontier_assignments(position: SolverPosition) -> Iterator[dict[Coordinate, bool]]:
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

        def backtrack(frontier_idx: int, moles_left: int) -> Iterator[dict[Coordinate, bool]]:
            if frontier_idx == len(frontier):
                # remaining moles must fit into deep unknowns
                if 0 <= moles_left <= deep_unknown_count:
                    yield assignment.copy()

                return

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
    def outcomes_for_move(position: SolverPosition, move: Coordinate) -> Iterator[tuple[bool, int | None]]:
        seen: set[tuple[bool, int | None]] = set()

        # We must look at ALL valid frontier assignments
        for assignment in Solver.cached_generate_frontier_assignments(position):
            if move in assignment:
                # Frontier cell logic
                is_mole = assignment[move]
                val = None
                if not is_mole:
                    val = sum(1 for n in Solver.neighbours(position.size, move.row, move.col)
                              if assignment.get(n) is True)

                outcome = (is_mole, val)
                if outcome not in seen:
                    seen.add(outcome)
                    yield outcome
            else:
                # Deep unknown cell logic:
                # It could be a mole or not, but we must check if
                # there are moles left to place there.
                frontier_set: set[Coordinate] = Solver.get_frontier(position)
                moles_in_frontier: int = sum(1 for is_mole in assignment.values() if is_mole)
                moles_left_for_deep: int = position.remaining_targets - moles_in_frontier

                deep_cells: list[Coordinate] = [Coordinate(r, c) for r in range(position.size) for c in range(position.size)
                              if position.board[r][c].hidden and Coordinate(r, c) not in frontier_set]

                # If we have moles left, 'True' is a possible outcome
                if moles_left_for_deep > 0:
                    if (True, None) not in seen:
                        seen.add((True, None))
                        yield True, None

                # If there's room to NOT be a mole, 'False' is possible
                if len(deep_cells) > moles_left_for_deep:
                    # For neighbor count, we calculate how many frontier moles
                    # touch this deep cell.
                    nearby_frontier_moles = sum(1 for n in Solver.neighbours(position.size, move.row, move.col)
                                                if assignment.get(n) is True)

                    # Note: In a perfect solver, you'd also consider moles placed
                    # in other deep cells, but usually, '0' or 'nearby_frontier_moles'
                    # is a safe lower bound.
                    outcome = (False, nearby_frontier_moles)
                    if outcome not in seen:
                        seen.add(outcome)
                        yield outcome

    # @staticmethod
    # def outcomes_for_move(position: SolverPosition, move: Coordinate) -> Iterator[tuple[bool, int | None]]:
    #     frontier: set[Coordinate] = Solver.get_frontier(position)
    #
    #     if move not in frontier:
    #         # deep unknown cell => unconstrained
    #         yield True, None
    #         yield False, 0
    #         return
    #
    #     # frontier cell:
    #     seen: set[tuple[bool, int]] = set()
    #     for assignment in Solver.cached_generate_frontier_assignments(position):
    #         if move not in assignment:
    #             continue
    #
    #         is_mole: bool = assignment[move]
    #         if is_mole:
    #             outcome = is_mole, None
    #         else:
    #             neighbour_moles: int = sum(1
    #                                        for neighbour in Solver.neighbours(position.size, move.row, move.col)
    #                                        if assignment.get(neighbour) is True)
    #             outcome = is_mole, neighbour_moles
    #
    #         if outcome not in seen:
    #             seen.add(outcome)
    #             yield outcome



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
                if neighbour.is_mole:
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
    def worst_case_remaining_shots(position: SolverPosition) -> int:
        if not Solver.position_is_consistent(position):
            return -1  # impossible path

        if position.remaining_targets == 0:
            return 0

        unrevealed_coordinates: list[Coordinate] = [Coordinate(row, col) for cell, row, col in
                                                    Solver.get_all_cells(position) if cell.hidden]
        if not unrevealed_coordinates:
            return Solver.INF  # this should never happen

        best_move_worst_case: int = Solver.INF

        move_found: bool = False
        for move in unrevealed_coordinates:
            possible_outcomes = list(Solver.outcomes_for_move(position, move))
            if not possible_outcomes:
                continue

            move_found = True
            current_move_max: int = 0

            for is_mole, neighbour_moles in possible_outcomes:
                next_pos: SolverPosition = Solver.apply_move(position, move, is_mole, neighbour_moles)

                next_step_result = Solver.worst_case_remaining_shots(next_pos)
                if next_step_result != -1:
                    current_move_max = max(current_move_max, 1 + next_step_result)

            if current_move_max > 0:
                best_move_worst_case = min(best_move_worst_case, current_move_max)

        return best_move_worst_case

    def calculate_best_field(self):
        Solver.worst_case_remaining_shots.cache_clear()
        Solver.cached_generate_frontier_assignments.cache_clear()

        best_minimax_cost: int = Solver.INF
        best_move: Coordinate | None = None

        unrevealed_coordinates: list[Coordinate] = [Coordinate(row, col) for cell, row, col in
                                                    Solver.get_all_cells(self.position) if cell.hidden]

        i: int = 0
        for cell, row, col in Solver.get_all_cells(self.position):
            if not cell.hidden:
                continue

            print(f"Trying {cell = } ({i}/{len(unrevealed_coordinates)})")
            i += 1

            move: Coordinate = Coordinate(row, col)
            candidate_move_worst_case_cost: int = 0
            possible_outcomes = list(Solver.outcomes_for_move(self.position, move))

            if not possible_outcomes:
                continue

            for is_mole, neighbour_moles in possible_outcomes:
                next_pos: SolverPosition = Solver.apply_move(self.position, move, is_mole, neighbour_moles)

                next_step_result = Solver.worst_case_remaining_shots(next_pos)
                if next_step_result != -1:
                    candidate_move_worst_case_cost = max(candidate_move_worst_case_cost, 1 + next_step_result)

            print(f"\t{candidate_move_worst_case_cost}")

            if 0 < candidate_move_worst_case_cost < best_minimax_cost:
                best_minimax_cost = candidate_move_worst_case_cost
                best_move = move

        self.best_move = best_move
        self.most_shots_necessary = best_minimax_cost

    def stop_calculation(self):
        pass
