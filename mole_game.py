import itertools
import random
import pygame
import math
import mole_solver


def flatten(*xss) -> list:
    return [x for xs in xss for x in (flatten(*xs) if isinstance(xs, (list, tuple, set)) else (xs,))]

def lighten(color: pygame.Color, factor: float) -> pygame.Color:
    return pygame.Color(
        int(color.r + (255 - color.r) * factor),
        int(color.g + (255 - color.g) * factor),
        int(color.b + (255 - color.b) * factor)
    )


class Field:
    is_revealed: bool
    has_mole: bool
    neighbours: int
    is_hovered: bool

    def __init__(self):
        self.is_revealed = False
        self.has_mole = False
        self.neighbours = 0
        self.is_hovered = False

    def __str__(self):
        if self.is_revealed:
            return 'X' if self.has_mole else f"{self.neighbours}"

        return '-'

    def __repr__(self):
        return 'X' if self.has_mole else f"{self.neighbours}"


class Game:
    _size: int  # the board is a size X size board
    _board: list[list[Field]]
    _number_of_targets: int
    _number_of_found_targets: int
    _number_of_revealed: int
    _solver: mole_solver.Solver

    def __init__(self, board_size: int, number_of_targets: int):
        assert board_size > 0 and board_size ** 2 >= number_of_targets > 0

        self._size = board_size
        self._number_of_targets = number_of_targets
        self._number_of_found_targets = 0
        self._number_of_revealed = 0
        self._solver = mole_solver.Solver(self._size, self._number_of_targets)

    def _reset_board(self):
        self._board = [[Field() for _ in range(self._size)] for _ in range(self._size)]
        self._number_of_found_targets = 0
        self._number_of_revealed = 0

        for field in random.sample(flatten(self._board), k=self._number_of_targets):
            field.has_mole = True

    def instantiate_board(self, fields: list[list[Field]] = None):
        if fields is None:
            self._reset_board()
        else:
            self._board = fields
            self._number_of_found_targets = sum(1 for field in flatten(fields) if field.is_revealed and field.has_mole)
            self._number_of_revealed = sum(1 for field in flatten(fields) if field.is_revealed)

        for row, data in enumerate(self._board):
            for column, field in enumerate(data):
                # if field.has_mole:
                #     continue

                neighbouring_fields_coordinates = [
                    coordinate for coordinate in
                    itertools.product([row - 1, row, row + 1], [column - 1, column, column + 1])
                    if (coordinate[0] != row or coordinate[1] != column)
                       and self._size > coordinate[0] >= 0
                       and self._size > coordinate[1] >= 0
                ]
                field.neighbours = sum(
                    self._board[neighbour_row][neighbour_column].has_mole
                    for neighbour_row, neighbour_column in neighbouring_fields_coordinates
                )

        return self

    def print_for_human_play(self):
        for row in self._board:
            print(*row)

    def print_xray(self):
        for row in self._board:
            print(row)

    def shoot_target(self, row: int, column: int) -> bool:
        assert self._size > row >= 0 and self._size > column >= 0

        if self._board[row][column].is_revealed:
            return False

        self._board[row][column].is_revealed = True
        self._number_of_revealed += 1
        self._number_of_found_targets += self._board[row][column].has_mole
        return True

    def is_finished(self):
        return self._number_of_found_targets == self._number_of_targets

    def start_engine(self):
        print("Starting engine...")
        self.stop_engine()
        self.solver.load_position(self)
        self.solver.calculate_best_field()
        print("Finished calculation")

    def stop_engine(self):
        self.solver.stop_calculation()

    @property
    def board(self):
        return self._board

    @property
    def number_of_found_targets(self):
        return self._number_of_found_targets

    @property
    def number_of_targets(self):
        return self._number_of_targets

    @property
    def number_of_revealed(self):
        return self._number_of_revealed

    @property
    def solver(self):
        return self._solver

    @property
    def size(self):
        return self._size


class Button:
    def __init__(self, rect, callback):
        self.rect = rect
        self.callback = callback

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.callback()


if __name__ == '__main__':
    # game settings
    BOARD_SIZE: int = 4
    MOLES: int = 5
    FONT = 'comicsans'
    engine_enabled: bool = False
    BUTTON_COLOR = "dodgerblue"
    BUTTON_HOVER_COLOR = "lightskyblue"

    # pixel values
    WINDOW_WIDTH: int = 900
    WINDOW_HEIGHT: int = 600
    FONT_SIZE: int = 30
    GAME_SIZE: int = min(WINDOW_WIDTH, WINDOW_HEIGHT)
    FIELD_SIZE: int = GAME_SIZE // BOARD_SIZE
    CIRCLE_RADIUS: int = int(FIELD_SIZE / math.pi)
    CLICK_TOLERANCE: int = int(math.log1p(CIRCLE_RADIUS))
    SIDEBAR_X_OFFSET: int = 15
    SIDEBAR_Y_OFFSET = 20

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    game = Game(BOARD_SIZE, MOLES)
    game.instantiate_board()


    def draw_game(mouse_pos) -> list[Button]:
        screen.fill('gray')

        # Main game
        for row, column in itertools.product(range(BOARD_SIZE), range(BOARD_SIZE)):
            x_pixel: int = row * FIELD_SIZE + FIELD_SIZE // 2
            y_pixel: int = column * FIELD_SIZE + FIELD_SIZE // 2
            pygame.draw.circle(screen, 'orange', (x_pixel, y_pixel), CIRCLE_RADIUS)

            if game.board[row][column].is_revealed:
                if game.board[row][column].has_mole:
                    pygame.draw.circle(screen, 'brown', (x_pixel, y_pixel), CIRCLE_RADIUS // 2)
                else:
                    circle_font = pygame.font.SysFont(FONT, CIRCLE_RADIUS)
                    text = circle_font.render(f"{game.board[row][column].neighbours}", True, (255, 255, 255))
                    text_rect = text.get_rect(center=(x_pixel, y_pixel))
                    screen.blit(text, text_rect)
            else:
                if game.board[row][column].is_hovered:
                    pygame.draw.circle(screen, lighten(pygame.Color('orange'), 0.25), (x_pixel, y_pixel), CIRCLE_RADIUS)

        interactive_buttons: list[Button] = []

        # Sidebar
        sidebar_font = pygame.font.SysFont(FONT, FONT_SIZE)

        moles_text = sidebar_font.render("Moles", True, (255, 255, 255))
        moles_rect = moles_text.get_rect(topleft=(GAME_SIZE, SIDEBAR_Y_OFFSET))
        moles_number_text = sidebar_font.render(f"{game.number_of_found_targets}/{game.number_of_targets}", True, (255, 255, 255))
        moles_number_rect = moles_number_text.get_rect(topright=(WINDOW_WIDTH - SIDEBAR_X_OFFSET, SIDEBAR_Y_OFFSET))

        attempts_text = sidebar_font.render(f"Shots taken", True, (255, 255, 255))
        attempts_rect = attempts_text.get_rect(topleft=(GAME_SIZE, SIDEBAR_Y_OFFSET + FONT_SIZE * 2))
        attempts_number_text = sidebar_font.render(f"{game.number_of_revealed}", True, (255, 255,255))
        attempts_number_rect = attempts_number_text.get_rect(topright=(WINDOW_WIDTH - SIDEBAR_X_OFFSET, SIDEBAR_Y_OFFSET + FONT_SIZE * 2))

        screen.blit(moles_text, moles_rect)
        screen.blit(moles_number_text, moles_number_rect)
        screen.blit(attempts_text, attempts_rect)
        screen.blit(attempts_number_text, attempts_number_rect)


        # Engine
        engine_button_text = sidebar_font.render(f" Engine {'(X)' if engine_enabled else '( )'} ", True, (255, 255, 255))
        engine_button_rect = engine_button_text.get_rect(topleft=(GAME_SIZE, SIDEBAR_Y_OFFSET + FONT_SIZE * 4))
        pygame.draw.rect(screen, BUTTON_HOVER_COLOR if engine_button_rect.collidepoint(mouse_pos) else BUTTON_COLOR, engine_button_rect)
        def toggle_engine():
            global engine_enabled
            engine_enabled = not engine_enabled

            if engine_enabled:
                game.start_engine()
            else:
                game.stop_engine()


        interactive_buttons.append(Button(engine_button_rect, toggle_engine))
        screen.blit(engine_button_text, engine_button_rect)

        if engine_enabled:
            depth_text = sidebar_font.render("Worst Case:", True, (255, 255, 255))
            depth_rect = depth_text.get_rect(topleft=(GAME_SIZE, SIDEBAR_Y_OFFSET + FONT_SIZE * 6))
            depth_value_text = sidebar_font.render(f"{game.solver.most_shots_necessary}", True, (255, 255, 255))
            depth_value_rect = depth_value_text.get_rect(topright=(WINDOW_WIDTH - SIDEBAR_X_OFFSET, SIDEBAR_Y_OFFSET + FONT_SIZE * 6))
            best_move_text = sidebar_font.render("Best Move:", True, (255, 255, 255))
            best_move_rect = best_move_text.get_rect(topleft=(GAME_SIZE, SIDEBAR_Y_OFFSET + FONT_SIZE * 8))
            best_move_value_text = sidebar_font.render(f"{game.solver.best_move}", True, (255, 255, 255))
            best_move_value_rect = best_move_value_text.get_rect(topright=(WINDOW_WIDTH - SIDEBAR_X_OFFSET, SIDEBAR_Y_OFFSET + FONT_SIZE * 8))

            screen.blit(depth_text, depth_rect)
            screen.blit(depth_value_text, depth_value_rect)
            screen.blit(best_move_text, best_move_rect)
            screen.blit(best_move_value_text, best_move_value_rect)


        if game.is_finished():
            success_text = sidebar_font.render("Success!", True, pygame.Color('red'))
            success_rect = success_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            screen.blit(success_text, success_rect)

        return interactive_buttons

    def get_row_and_col_from_pixel(position: tuple[int, int]) -> tuple[int, int] | None:
        x_px, y_px = position
        row = x_px // FIELD_SIZE
        column = y_px // FIELD_SIZE
        return row, column

    def get_field_from_pixel(position: tuple[int, int]) -> Field | None:
        x_px, y_px = position
        if not (0 <= x_px < GAME_SIZE and 0 <= y_px < GAME_SIZE):
            return None

        row, column = get_row_and_col_from_pixel(position)
        accept_click_radius: int = CIRCLE_RADIUS + CLICK_TOLERANCE
        center_x_px: int = row * FIELD_SIZE + FIELD_SIZE // 2
        center_y_px: int = column * FIELD_SIZE + FIELD_SIZE // 2
        distance_x_px: int = x_px - center_x_px
        distance_y_px: int = y_px - center_y_px
        distance_px_squared: int = distance_x_px ** 2 + distance_y_px ** 2

        if distance_px_squared > accept_click_radius ** 2:
            return None

        return game.board[row][column]


    interactive_buttons: list[Button] = []
    while running:
        for event in pygame.event.get():
            for button in interactive_buttons:
                button.handle_event(event)

            if event.type == pygame.QUIT:
                running = False

            if game.is_finished():
                # ignore everything except above
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                field: Field = get_field_from_pixel(event.pos)
                if field is None:
                    continue


                if game.shoot_target(*get_row_and_col_from_pixel(event.pos)) and engine_enabled:
                    game.start_engine()

                game.print_for_human_play()
                print()

            if event.type == pygame.MOUSEMOTION:
                for field in flatten(game.board):
                    field.is_hovered = False

                field: Field = get_field_from_pixel(event.pos)
                if field is None or field.is_revealed:
                    continue

                field.is_hovered = True

        interactive_buttons = draw_game(pygame.mouse.get_pos())
        pygame.display.update()
        clock.tick(60)
