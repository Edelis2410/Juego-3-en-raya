import pygame
import sys
import math
import os

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PRIMARY_BLUE = (25, 130, 196)
DARK_BLUE = (15, 70, 130)
SECONDARY_BLUE = (70, 130, 180)
ACCENT_YELLOW = (255, 200, 50)
PLAYER_X_COLOR = (220, 60, 60)
PLAYER_O_COLOR = (60, 130, 220)
LIGHT_GRAY = (240, 245, 250)
MEDIUM_GRAY = (200, 210, 220)
DARK_GRAY = (40, 50, 65)
BOARD_LINES = (0, 0, 0)

BACKGROUND_DARK = (25, 15, 55)
BACKGROUND_LIGHT = (45, 25, 85)

LINE_WIDTH = 8


WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800

# Tablero principal
BOARD_SIZE = 3
CELL_SIZE = 130
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE

# Centrar el tablero en el lado izquierdo
LEFT_PANEL_WIDTH = 520
BOARD_OFFSET_X = (LEFT_PANEL_WIDTH - BOARD_WIDTH) // 2
BOARD_OFFSET_Y = (WINDOW_HEIGHT - BOARD_HEIGHT) // 2

# Mini tableros para el grafo
MINI_CELL_SIZE = 20
MINI_BOARD_SIZE = MINI_CELL_SIZE * 3

font_title = pygame.font.SysFont('Arial Black', 48, bold=True)
font_subtitle = pygame.font.SysFont('Arial', 28, bold=True)
font_medium = pygame.font.SysFont('Arial', 24)
font_small = pygame.font.SysFont('Arial', 20)
font_tiny = pygame.font.SysFont('Arial', 14)


class GameState:
    def __init__(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        self.nodes_evaluated = 0
        self.move_history = []

    def reset(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        self.nodes_evaluated = 0
        self.move_history = []

    def make_move(self, row, col, player):
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            self.move_history.append((row, col, player))
            return True
        return False

    def copy(self):
        new_state = GameState()
        new_state.board = [row[:] for row in self.board]
        new_state.current_player = self.current_player
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        return new_state

    def check_winner(self):
        # Filas
        for row in range(BOARD_SIZE):
            if self.board[row][0] != ' ' and self.board[row][0] == self.board[row][1] == self.board[row][2]:
                return self.board[row][0]
        # Columnas
        for col in range(BOARD_SIZE):
            if self.board[0][col] != ' ' and self.board[0][col] == self.board[1][col] == self.board[2][col]:
                return self.board[0][col]
        # Diagonales
        if self.board[0][0] != ' ' and self.board[0][0] == self.board[1][1] == self.board[2][2]:
            return self.board[0][0]
        if self.board[0][2] != ' ' and self.board[0][2] == self.board[1][1] == self.board[2][0]:
            return self.board[0][2]
        # Empate
        if all(self.board[row][col] != ' ' for row in range(BOARD_SIZE) for col in range(BOARD_SIZE)):
            return 'Tie'
        return None

    def get_empty_cells(self):
        return [(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE)
                if self.board[row][col] == ' ']

    def is_terminal(self):
        return self.check_winner() is not None


class MinimaxAgent:
    def __init__(self):
        self.nodes_visited = 0
        self.best_move = None
        self.decision_tree = []

    def reset(self):
        self.nodes_visited = 0
        self.decision_tree = []

    def boards_equal(self, board1, board2):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board1[r][c] != board2[r][c]:
                    return False
        return True

    def find_matching_node_recursive(self, node, target_board):
        if not node:
            return None

        if node['board'] and self.boards_equal(node['board'], target_board):
            return node

        if 'children' in node:
            for child in node['children']:
                found = self.find_matching_node_recursive(child, target_board)
                if found:
                    return found

        return None

    def evaluate(self, state):
        winner = state.check_winner()
        if winner == 'O':
            return 1
        elif winner == 'X':
            return -1
        elif winner == 'Tie':
            return 0

        score = 0
        for row in range(3):
            score += self.evaluate_line([state.board[row][0],
                                         state.board[row][1],
                                         state.board[row][2]])
        for col in range(3):
            score += self.evaluate_line([state.board[0][col],
                                         state.board[1][col],
                                         state.board[2][col]])
        score += self.evaluate_line([state.board[0][0],
                                     state.board[1][1],
                                     state.board[2][2]])
        score += self.evaluate_line([state.board[0][2],
                                     state.board[1][1],
                                     state.board[2][0]])
        return score

    def evaluate_line(self, line):
        if line.count('O') == 2 and line.count(' ') == 1:
            return 5
        elif line.count('X') == 2 and line.count(' ') == 1:
            return -5
        elif line.count('O') == 1 and line.count(' ') == 2:
            return 1
        elif line.count('X') == 1 and line.count(' ') == 2:
            return -1
        return 0

    def minimax(self, state, depth, maximizing_player, node):
        self.nodes_visited += 1
        node['board'] = [row[:] for row in state.board]

        if state.is_terminal():
            score = self.evaluate(state)
            node['score'] = score
            node['terminal'] = True
            return score

        max_depth_to_draw = 2

        if maximizing_player:
            best_val = -math.inf
            best_move = None
            for (row, col) in state.get_empty_cells():
                new_state = state.copy()
                new_state.board[row][col] = 'O'

                child_node = {
                    'board': None,
                    'depth': depth + 1,
                    'maximizing': False,
                    'score': None,
                    'children': [],
                    'move': (row, col)
                }
                if depth <= max_depth_to_draw:
                    node['children'].append(child_node)

                val = self.minimax(new_state, depth + 1, False, child_node)

                if val > best_val:
                    best_val = val
                    best_move = (row, col)

            node['score'] = best_val
            if depth == 0:
                node['best_move'] = best_move
                self.best_move = best_move
            return best_val
        else:
            best_val = math.inf
            for (row, col) in state.get_empty_cells():
                new_state = state.copy()
                new_state.board[row][col] = 'X'

                child_node = {
                    'board': None,
                    'depth': depth + 1,
                    'maximizing': True,
                    'score': None,
                    'children': [],
                    'move': (row, col)
                }
                if depth <= max_depth_to_draw:
                    node['children'].append(child_node)

                val = self.minimax(new_state, depth + 1, True, child_node)

                if val < best_val:
                    best_val = val

            node['score'] = best_val
            return best_val

    def get_best_move(self, state):
        self.nodes_visited = 0
        self.best_move = None

        if not self.decision_tree:
            root = {
                'board': [row[:] for row in state.board],
                'depth': 0,
                'maximizing': True,
                'score': None,
                'children': [],
                'move': None
            }
            self.decision_tree.append(root)
        else:
            root = self.decision_tree[0]

            matching_node = self.find_matching_node_recursive(root, state.board)

            if matching_node and matching_node != root:
                self.decision_tree[0] = matching_node
            else:
                root['board'] = [row[:] for row in state.board]

        self.minimax(state, 0, True, self.decision_tree[0])

        return self.best_move if self.best_move else state.get_empty_cells()[0]


class GameGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tres en Raya - Árbol de Decisiones")
        self.clock = pygame.time.Clock()
        self.state = GameState()
        self.agent = MinimaxAgent()

        # Cargar imágenes
        self.images_loaded = False
        try:
            self.board_image = pygame.image.load('tablero.png').convert_alpha()
            self.x_image = pygame.image.load('equis.png').convert_alpha()
            self.o_image = pygame.image.load('circle.png').convert_alpha()
            self.images_loaded = True

            self.board_image = pygame.transform.smoothscale(self.board_image, (BOARD_WIDTH, BOARD_HEIGHT))
            cell_content_size = CELL_SIZE - 20
            self.x_image = pygame.transform.smoothscale(self.x_image, (cell_content_size, cell_content_size))
            self.o_image = pygame.transform.smoothscale(self.o_image, (cell_content_size, cell_content_size))

        except:
            print("Sin imágenes - usando tablero clásico")

        button_width = 220
        button_x = BOARD_OFFSET_X + (BOARD_WIDTH - button_width) // 2
        self.new_game_button = pygame.Rect(button_x, BOARD_OFFSET_Y + BOARD_HEIGHT + 60, button_width, 60)

    def draw_dark_gradient_background(self):
    
        for i in range(WINDOW_HEIGHT):
            ratio = i / WINDOW_HEIGHT
            color = (
                int(BACKGROUND_DARK[0] * (1 - ratio) + BACKGROUND_LIGHT[0] * ratio),
                int(BACKGROUND_DARK[1] * (1 - ratio) + BACKGROUND_LIGHT[1] * ratio),
                int(BACKGROUND_DARK[2] * (1 - ratio) + BACKGROUND_LIGHT[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, i), (WINDOW_WIDTH, i))

    def draw_classic_board(self):
        board_rect = pygame.Rect(BOARD_OFFSET_X - 15, BOARD_OFFSET_Y - 15, BOARD_WIDTH + 30, BOARD_HEIGHT + 30)
        pygame.draw.rect(self.screen, WHITE, board_rect, border_radius=20)
        pygame.draw.rect(self.screen, BLACK, board_rect, 4, border_radius=20)

        line_width = LINE_WIDTH

        # Líneas verticales
        for i in range(1, BOARD_SIZE):
            x_pos = BOARD_OFFSET_X + i * CELL_SIZE
            pygame.draw.line(self.screen, BOARD_LINES,
                             (x_pos, BOARD_OFFSET_Y),
                             (x_pos, BOARD_OFFSET_Y + BOARD_HEIGHT), line_width)

        # Líneas horizontales
        for i in range(1, BOARD_SIZE):
            y_pos = BOARD_OFFSET_Y + i * CELL_SIZE
            pygame.draw.line(self.screen, BOARD_LINES,
                             (BOARD_OFFSET_X, y_pos),
                             (BOARD_OFFSET_X + BOARD_WIDTH, y_pos), line_width)

        # Fichas
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.state.board[row][col] != ' ':
                    cell_x = BOARD_OFFSET_X + col * CELL_SIZE + (CELL_SIZE // 2)
                    cell_y = BOARD_OFFSET_Y + row * CELL_SIZE + (CELL_SIZE // 2)

                    if self.images_loaded and self.state.board[row][col] == 'X':
                        img_x = BOARD_OFFSET_X + col * CELL_SIZE + 10
                        img_y = BOARD_OFFSET_Y + row * CELL_SIZE + 10
                        self.screen.blit(self.x_image, (img_x, img_y))
                    elif self.images_loaded and self.state.board[row][col] == 'O':
                        img_x = BOARD_OFFSET_X + col * CELL_SIZE + 10
                        img_y = BOARD_OFFSET_Y + row * CELL_SIZE + 10
                        self.screen.blit(self.o_image, (img_x, img_y))
                    else:
                        if self.state.board[row][col] == 'X':
                            pygame.draw.line(self.screen, PLAYER_X_COLOR,
                                             (cell_x - 30, cell_y - 30),
                                             (cell_x + 30, cell_y + 30), 12)
                            pygame.draw.line(self.screen, PLAYER_X_COLOR,
                                             (cell_x + 30, cell_y - 30),
                                             (cell_x - 30, cell_y + 30), 12)
                        elif self.state.board[row][col] == 'O':
                            pygame.draw.circle(self.screen, PLAYER_O_COLOR, (cell_x, cell_y), 35, 10)

        title_shadow = font_title.render("TRES EN RAYA", True, BLACK)
        title = font_title.render("TRES EN RAYA", True, ACCENT_YELLOW)
        title_x = BOARD_OFFSET_X + (BOARD_WIDTH - title.get_width()) // 2
        self.screen.blit(title_shadow, (title_x + 3, BOARD_OFFSET_Y - 77))
        self.screen.blit(title, (title_x, BOARD_OFFSET_Y - 80))

    def draw_mini_board(self, board, x, y, score=None, highlight=False, is_current=False, depth=0, is_best=False):
        mini_board_rect = pygame.Rect(x, y, MINI_BOARD_SIZE, MINI_BOARD_SIZE)

        if is_current:
            pygame.draw.rect(self.screen, ACCENT_YELLOW, mini_board_rect, border_radius=4)
            pygame.draw.rect(self.screen, BLACK, mini_board_rect, 2, border_radius=4)
        elif is_best:
            pygame.draw.rect(self.screen, (200, 220, 255), mini_board_rect, border_radius=4)
            pygame.draw.rect(self.screen, PLAYER_O_COLOR, mini_board_rect, 2, border_radius=4)
        elif highlight:
            pygame.draw.rect(self.screen, (245, 245, 245), mini_board_rect, border_radius=3)
        else:
            pygame.draw.rect(self.screen, WHITE, mini_board_rect, border_radius=3)
            pygame.draw.rect(self.screen, BLACK, mini_board_rect, 1, border_radius=3)

        for i in range(1, 3):
            pygame.draw.line(self.screen, BLACK,
                             (x + i * MINI_CELL_SIZE, y + 2),
                             (x + i * MINI_CELL_SIZE, y + MINI_BOARD_SIZE - 2), 1)
            pygame.draw.line(self.screen, BLACK,
                             (x + 2, y + i * MINI_CELL_SIZE),
                             (x + MINI_BOARD_SIZE - 2, y + i * MINI_CELL_SIZE), 1)

        for row in range(3):
            for col in range(3):
                cell_x = x + col * MINI_CELL_SIZE + 2
                cell_y = y + row * MINI_CELL_SIZE + 2

                if board[row][col] == 'X':
                    pygame.draw.line(self.screen, PLAYER_X_COLOR,
                                     (cell_x + 1, cell_y + 1),
                                     (cell_x + MINI_CELL_SIZE - 3, cell_y + MINI_CELL_SIZE - 3), 2)
                    pygame.draw.line(self.screen, PLAYER_X_COLOR,
                                     (cell_x + MINI_CELL_SIZE - 3, cell_y + 1),
                                     (cell_x + 1, cell_y + MINI_CELL_SIZE - 3), 2)
                elif board[row][col] == 'O':
                    pygame.draw.circle(self.screen, PLAYER_O_COLOR,
                                       (cell_x + MINI_CELL_SIZE // 2, cell_y + MINI_CELL_SIZE // 2),
                                       MINI_CELL_SIZE // 2 - 3, 2)

        if score is not None:
            score_color = PLAYER_O_COLOR if score >= 0 else PLAYER_X_COLOR
            score_text = font_tiny.render(f"{score:+d}", True, score_color)
            self.screen.blit(score_text, (x + MINI_BOARD_SIZE + 5, y + 2))

    def draw_decision_tree(self):
        graph_x = 580
        graph_y = BOARD_OFFSET_Y
        graph_width = WINDOW_WIDTH - graph_x - 50
        graph_height = BOARD_HEIGHT + 100

        panel_rect = pygame.Rect(graph_x - 20, graph_y - 15, graph_width + 40, graph_height + 30)
        pygame.draw.rect(self.screen, WHITE, panel_rect, border_radius=20)
        pygame.draw.rect(self.screen, PRIMARY_BLUE, panel_rect, 3, border_radius=20)

        title_shadow = font_title.render("ÁRBOL DE DECISIONES", True, BLACK)
        title = font_title.render("ÁRBOL DE DECISIONES", True, WHITE)
        title_x = panel_rect.centerx - title.get_width() // 2
        self.screen.blit(title_shadow, (title_x + 3, graph_y - 77))
        self.screen.blit(title, (title_x, graph_y - 80))

        if self.agent.nodes_visited > 0:
            stats_rect = pygame.Rect(graph_x + 10, graph_y - 10, 350, 50)
            pygame.draw.rect(self.screen, LIGHT_GRAY, stats_rect, border_radius=10)
            pygame.draw.rect(self.screen, SECONDARY_BLUE, stats_rect, 2, border_radius=10)

            stats_text = font_small.render(f"Nodos evaluados: {self.agent.nodes_visited}", True, BLACK)
            self.screen.blit(stats_text, (graph_x + 25, graph_y + 5))
        else:
            stats_text = font_small.render("Esperando movimiento...", True, DARK_GRAY)
            self.screen.blit(stats_text, (graph_x + 25, graph_y + 5))

        self.draw_legend_in_tree_panel(panel_rect)

        if self.agent.decision_tree:
            self.draw_tree_structure(self.agent.decision_tree, graph_x + 50, graph_y + 60, graph_width - 100)
        else:
            no_tree = font_medium.render("Haz clic en el tablero para ver el análisis", True, DARK_GRAY)
            self.screen.blit(no_tree, (graph_x + graph_width // 2 - no_tree.get_width() // 2, graph_y + 150))

    def draw_legend_in_tree_panel(self, panel_rect):
        legend_width = 300
        legend_height = 40
        legend_x = panel_rect.x + 20
        legend_y = panel_rect.y + panel_rect.height - legend_height - 20

        legend_rect = pygame.Rect(legend_x, legend_y, legend_width, legend_height)
        pygame.draw.rect(self.screen, WHITE, legend_rect, border_radius=12)

        legend_items = [
            (ACCENT_YELLOW, "Nodo Raíz"),
            (PLAYER_O_COLOR, "Mejor Movimiento")
        ]

        start_x = legend_x + 15
        start_y = legend_y + (legend_height - 14) // 2
        item_spacing = 140

        for i, (color, text) in enumerate(legend_items):
            x_pos = start_x + i * item_spacing

            pygame.draw.rect(self.screen, color, (x_pos, start_y, 14, 14), border_radius=3)

            text_surface = font_tiny.render(text, True, BLACK)
            self.screen.blit(text_surface, (x_pos + 20, start_y - 2))

    def draw_tree_structure(self, nodes, start_x, start_y, available_width):
        if not nodes:
            return

        level_height = 130
        node = nodes[0] if isinstance(nodes, list) else nodes
        x = start_x + available_width // 2
        y = start_y

        self.draw_mini_board(node['board'], x - MINI_BOARD_SIZE // 2, y, node.get('score'), is_current=True, depth=0)

        if node.get('best_move'):
            row, col = node['best_move']
            best_text = font_tiny.render(f"Mejor: ({row},{col})", True, BLACK)
            bg_rect = pygame.Rect(x - MINI_BOARD_SIZE // 2 - 5, y + MINI_BOARD_SIZE + 5,
                                  best_text.get_width() + 10, best_text.get_height() + 5)
            pygame.draw.rect(self.screen, (200, 230, 255), bg_rect, border_radius=3)
            self.screen.blit(best_text, (x - MINI_BOARD_SIZE // 2, y + MINI_BOARD_SIZE + 8))

        if 'children' in node and node['children']:
            children = node['children'][:8]
            child_spacing = min(available_width // max(len(children), 1), 140)
            child_start_x = x - ((len(children) - 1) * child_spacing) // 2

            for i, child in enumerate(children):
                child_x = child_start_x + i * child_spacing
                child_y = y + level_height

                pygame.draw.line(self.screen, SECONDARY_BLUE,
                                 (x, y + MINI_BOARD_SIZE), (child_x + MINI_BOARD_SIZE // 2, child_y), 2)

                is_best = node.get('best_move') == child.get('move')
                self.draw_mini_board(child['board'], child_x - MINI_BOARD_SIZE // 2, child_y,
                                     child.get('score'), is_best=is_best, depth=1)

    def draw_simple_button(self, rect, text, color, text_color=WHITE):
        pygame.draw.rect(self.screen, color, rect, border_radius=12)
        pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=12)

        text_surface = font_medium.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def draw_new_game_button(self):
        self.draw_simple_button(self.new_game_button, "NUEVO JUEGO", (30, 160, 80), WHITE)

        if self.state.game_over:
            result_y = self.new_game_button.y + 100

            if self.state.winner == 'X':
                result_text = "¡VICTORIA HUMANA!"
                result_color = PLAYER_X_COLOR
            elif self.state.winner == 'O':
                result_text = "VICTORIA DE LA IA"
                result_color = PLAYER_O_COLOR
            else:
                result_text = "EMPATE"
                result_color = ACCENT_YELLOW

            result_shadow = font_subtitle.render(result_text, True, BLACK)
            result_surface = font_subtitle.render(result_text, True, result_color)
            result_rect = result_surface.get_rect(center=(BOARD_OFFSET_X + BOARD_WIDTH // 2, result_y))

            self.screen.blit(result_shadow, (result_rect.x + 3, result_rect.y + 3))
            self.screen.blit(result_surface, result_rect)

    def draw_close_button(self):
        button_rect = pygame.Rect(WINDOW_WIDTH - 60, 20, 40, 40)
        pygame.draw.rect(self.screen, PLAYER_X_COLOR, button_rect, border_radius=8)
        pygame.draw.rect(self.screen, WHITE, button_rect, 2, border_radius=8)

        x_text = font_medium.render("X", True, WHITE)
        x_rect = x_text.get_rect(center=button_rect.center)
        self.screen.blit(x_text, x_rect)

    def get_cell_from_pos(self, pos):
        x, y = pos
        if (BOARD_OFFSET_X <= x <= BOARD_OFFSET_X + BOARD_WIDTH and
                BOARD_OFFSET_Y <= y <= BOARD_OFFSET_Y + BOARD_HEIGHT):
            col = (x - BOARD_OFFSET_X) // CELL_SIZE
            row = (y - BOARD_OFFSET_Y) // CELL_SIZE
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                return row, col
        return None

    def run(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if (not self.state.game_over and self.state.current_player == 'X'):
                        cell = self.get_cell_from_pos(mouse_pos)
                        if cell and self.state.make_move(*cell, 'X'):
                            self.state.winner = self.state.check_winner()
                            if self.state.winner:
                                self.state.game_over = True
                            else:
                                self.state.current_player = 'O'

                    if self.new_game_button.collidepoint(mouse_pos):
                        self.state.reset()
                        self.agent.reset()

                    close_button = pygame.Rect(WINDOW_WIDTH - 60, 20, 40, 40)
                    if close_button.collidepoint(mouse_pos):
                        running = False

            self.draw_dark_gradient_background()
            pygame.draw.line(self.screen, PRIMARY_BLUE, (520, BOARD_OFFSET_Y - 60),
                             (520, BOARD_OFFSET_Y + BOARD_HEIGHT + 60), 3)
            self.draw_classic_board()
            self.draw_decision_tree()
            self.draw_new_game_button()
            self.draw_close_button()

            if (not self.state.game_over and self.state.current_player == 'O'):
                thinking_text = font_medium.render("IA pensando...", True, ACCENT_YELLOW)
                text_rect = thinking_text.get_rect(center=(BOARD_OFFSET_X + BOARD_WIDTH // 2,
                                                           BOARD_OFFSET_Y + BOARD_HEIGHT + 40))
                self.screen.blit(thinking_text, text_rect)

                pygame.display.flip()
                pygame.time.wait(500)

                best_move = self.agent.get_best_move(self.state)
                if best_move:
                    row, col = best_move
                    self.state.make_move(row, col, 'O')
                    self.state.winner = self.state.check_winner()
                    if self.state.winner:
                        self.state.game_over = True
                    else:
                        self.state.current_player = 'X'

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = GameGUI()
    game.run()
