import pygame
import sys
import math

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
CELL_SIZE = 90
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE

# Panel izquierdo
LEFT_PANEL_WIDTH = 380
BOARD_OFFSET_X = (LEFT_PANEL_WIDTH - BOARD_WIDTH) // 2
BOARD_OFFSET_Y = (WINDOW_HEIGHT - BOARD_HEIGHT) // 2

# Mini tableros para el grafo
MINI_CELL_SIZE = 16
MINI_BOARD_SIZE = MINI_CELL_SIZE * 3

font_title = pygame.font.SysFont('Arial Black', 38, bold=True)
font_subtitle = pygame.font.SysFont('Arial', 24, bold=True)
font_medium = pygame.font.SysFont('Arial', 20)
font_small = pygame.font.SysFont('Arial', 16)
font_tiny = pygame.font.SysFont('Arial', 12)
font_micro = pygame.font.SysFont('Arial', 10)
font_legend = pygame.font.SysFont('Arial', 16)


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
        self.max_tree_depth = 6 
        self.max_children_display = 4

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
        if winner == 'O':  # Victoria de la IA
            return 1
        elif winner == 'X':  # Victoria del jugador humano
            return -1
        elif winner == 'Tie':  # Empate
            return 0
        return None  # No hay evaluación para estados no terminales

    def minimax(self, state, depth, maximizing_player, node, max_depth=6):
        self.nodes_visited += 1
        node['board'] = [row[:] for row in state.board]
        node['depth'] = depth
        node['maximizing'] = maximizing_player
        node['terminal'] = False
        node['x'] = 0
        node['y'] = 0
        node['mod'] = 0
        node['thread'] = None
        node['offset'] = 0

        # Solo evaluamos si el estado es terminal
        if state.is_terminal():
            score = self.evaluate(state)
            node['score'] = score
            node['terminal'] = True
            return score
        
        # Si alcanzamos la profundidad máxima, continuamos explorando hasta terminales
        # pero limitamos la visualización
        if depth >= max_depth:
            # No retornamos valor, continuamos explorando pero con menos hijos para visualización
            pass

        if maximizing_player:
            best_val = -math.inf
            best_move = None
            empty_cells = state.get_empty_cells()

            # Ordenar movimientos para explorar primero los más prometedores
            # Simplemente usamos un orden central primero
            center_first = []
            corners = []
            edges = []
            for (row, col) in empty_cells:
                if (row, col) == (1, 1):
                    center_first.append((row, col))
                elif (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                    corners.append((row, col))
                else:
                    edges.append((row, col))
            
            ordered_moves = center_first + corners + edges
            limited_moves = ordered_moves[:self.max_children_display]

            for (row, col) in limited_moves:
                new_state = state.copy()
                new_state.board[row][col] = 'O'

                child_node = {
                    'board': None,
                    'depth': depth + 1,
                    'maximizing': False,
                    'score': None,
                    'children': [],
                    'move': (row, col),
                    'terminal': False,
                    'x': 0,
                    'y': 0,
                    'mod': 0,
                    'thread': None,
                    'offset': 0
                }
                node['children'].append(child_node)

                val = self.minimax(new_state, depth + 1, False, child_node, max_depth)

                # Solo actualizamos si val no es None
                if val is not None and val > best_val:
                    best_val = val
                    best_move = (row, col)

            # Si no encontramos valores, continuamos explorando
            if best_val == -math.inf:
                # Continuamos la búsqueda pero con un valor neutral temporal
                return 0
                
            node['score'] = best_val
            if depth == 0:
                node['best_move'] = best_move
                self.best_move = best_move
            return best_val
        else:
            best_val = math.inf
            empty_cells = state.get_empty_cells()

            # Mismo ordenamiento para MIN
            center_first = []
            corners = []
            edges = []
            for (row, col) in empty_cells:
                if (row, col) == (1, 1):
                    center_first.append((row, col))
                elif (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                    corners.append((row, col))
                else:
                    edges.append((row, col))
            
            ordered_moves = center_first + corners + edges
            limited_moves = ordered_moves[:self.max_children_display]

            for (row, col) in limited_moves:
                new_state = state.copy()
                new_state.board[row][col] = 'X'

                child_node = {
                    'board': None,
                    'depth': depth + 1,
                    'maximizing': True,
                    'score': None,
                    'children': [],
                    'move': (row, col),
                    'terminal': False,
                    'x': 0,
                    'y': 0,
                    'mod': 0,
                    'thread': None,
                    'offset': 0
                }
                node['children'].append(child_node)

                val = self.minimax(new_state, depth + 1, True, child_node, max_depth)

                if val is not None and val < best_val:
                    best_val = val

            if best_val == math.inf:
                return 0
                
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
                'move': None,
                'terminal': False,
                'x': 0,
                'y': 0,
                'mod': 0,
                'thread': None,
                'offset': 0
            }
            self.decision_tree.append(root)
        else:
            root = self.decision_tree[0]
            matching_node = self.find_matching_node_recursive(root, state.board)
            if matching_node and matching_node != root:
                self.decision_tree[0] = matching_node
            else:
                root['board'] = [row[:] for row in state.board]

        # Usamos un enfoque iterativo: primero intentamos con profundidad completa
        
        empty_cells = len(state.get_empty_cells())
        if empty_cells <= 6:  # Si quedan pocos movimientos, busca exhaustivo
            self.max_tree_depth = 9
        else:
            self.max_tree_depth = 4

        self.minimax(state, 0, True, self.decision_tree[0], self.max_tree_depth)

        # Si no encontramos un mejor movimiento, usamos una estrategia simple
        if not self.best_move:
            # Estrategia básica: centro > esquinas > bordes
            empty_cells = state.get_empty_cells()
            if (1, 1) in empty_cells:
                self.best_move = (1, 1)
            else:
                corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
                available_corners = [c for c in corners if c in empty_cells]
                if available_corners:
                    self.best_move = available_corners[0]
                elif empty_cells:
                    self.best_move = empty_cells[0]
                    
        return self.best_move


class TreeLayout:
    def __init__(self, horizontal_spacing=110, vertical_spacing=150):
        self.horizontal_spacing = horizontal_spacing
        self.vertical_spacing = vertical_spacing

    def calculate_positions(self, node, start_x, start_y, max_depth=3):
        if not node:
            return
        self.first_walk(node, 0)
        self.second_walk(node, 0, 0)
        self.normalize_positions(node, start_x, start_y)

    def first_walk(self, node, depth):
        if not node.get('children') or depth >= 3:
            node['x'] = 0
            node['mod'] = 0
            return

        for child in node['children']:
            self.first_walk(child, depth + 1)

        children = node['children']
        if len(children) == 1:
            children[0]['x'] = 0
        else:
            for i, child in enumerate(children):
                child['x'] = i * self.horizontal_spacing - ((len(children) - 1) * self.horizontal_spacing) / 2

    def second_walk(self, node, mod_sum, depth):
        if depth > 3:
            return

        node['x'] += mod_sum
        node['mod'] += mod_sum

        if node.get('children'):
            for child in node['children']:
                self.second_walk(child, node['mod'], depth + 1)

    def normalize_positions(self, node, start_x, start_y):
        min_x = float('inf')
        max_x = float('-inf')

        def find_bounds(n, depth):
            nonlocal min_x, max_x
            if depth > 3:
                return
            min_x = min(min_x, n['x'])
            max_x = max(max_x, n['x'])
            if n.get('children'):
                for child in n['children']:
                    find_bounds(child, depth + 1)

        find_bounds(node, 0)

        total_width = max_x - min_x
        if total_width == 0:
            total_width = 1

        def apply_positions(n, depth):
            if depth > 3:
                return
            normalized_x = (n['x'] - min_x) / total_width if total_width > 0 else 0.5
            n['screen_x'] = start_x + normalized_x * (WINDOW_WIDTH - LEFT_PANEL_WIDTH - 100)
            n['screen_y'] = start_y + depth * self.vertical_spacing
            if n.get('children'):
                for child in n['children']:
                    apply_positions(child, depth + 1)

        apply_positions(node, 0)


class GameGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tres en Raya - Árbol de Decisiones Minimax")
        self.clock = pygame.time.Clock()
        self.state = GameState()
        self.agent = MinimaxAgent()
        self.tree_layout = TreeLayout(horizontal_spacing=110, vertical_spacing=150)

        button_width = 180
        button_x = BOARD_OFFSET_X + (BOARD_WIDTH - button_width) // 2
        self.new_game_button = pygame.Rect(button_x, BOARD_OFFSET_Y + BOARD_HEIGHT + 40, button_width, 50)

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
        board_rect = pygame.Rect(BOARD_OFFSET_X - 10, BOARD_OFFSET_Y - 10, BOARD_WIDTH + 20, BOARD_HEIGHT + 20)
        pygame.draw.rect(self.screen, WHITE, board_rect, border_radius=15)
        pygame.draw.rect(self.screen, BLACK, board_rect, 3, border_radius=15)

        line_width = 6

        for i in range(1, BOARD_SIZE):
            x_pos = BOARD_OFFSET_X + i * CELL_SIZE
            pygame.draw.line(self.screen, BOARD_LINES,
                             (x_pos, BOARD_OFFSET_Y),
                             (x_pos, BOARD_OFFSET_Y + BOARD_HEIGHT), line_width)

        for i in range(1, BOARD_SIZE):
            y_pos = BOARD_OFFSET_Y + i * CELL_SIZE
            pygame.draw.line(self.screen, BOARD_LINES,
                             (BOARD_OFFSET_X, y_pos),
                             (BOARD_OFFSET_X + BOARD_WIDTH, y_pos), line_width)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.state.board[row][col] != ' ':
                    cell_x = BOARD_OFFSET_X + col * CELL_SIZE + (CELL_SIZE // 2)
                    cell_y = BOARD_OFFSET_Y + row * CELL_SIZE + (CELL_SIZE // 2)

                    if self.state.board[row][col] == 'X':
                        pygame.draw.line(self.screen, PLAYER_X_COLOR,
                                         (cell_x - 25, cell_y - 25),
                                         (cell_x + 25, cell_y + 25), 10)
                        pygame.draw.line(self.screen, PLAYER_X_COLOR,
                                         (cell_x + 25, cell_y - 25),
                                         (cell_x - 25, cell_y + 25), 10)
                    elif self.state.board[row][col] == 'O':
                        pygame.draw.circle(self.screen, PLAYER_O_COLOR, (cell_x, cell_y), 30, 8)

        title_shadow = font_title.render("TRES EN RAYA", True, BLACK)
        title = font_title.render("TRES EN RAYA", True, ACCENT_YELLOW)
        title_x = BOARD_OFFSET_X + (BOARD_WIDTH - title.get_width()) // 2
        self.screen.blit(title_shadow, (title_x + 2, BOARD_OFFSET_Y - 60))
        self.screen.blit(title, (title_x, BOARD_OFFSET_Y - 62))

    def draw_mini_board(self, board, x, y, score=None, node_type="MAX",
                        is_current=False, is_terminal=False, is_root=False):
        mini_board_rect = pygame.Rect(x, y, MINI_BOARD_SIZE, MINI_BOARD_SIZE)

        if is_current:
            bg_color = ACCENT_YELLOW
            border_color = BLACK
            border_width = 3
        elif is_terminal:
            bg_color = WHITE
            border_color = (255, 200, 0)
            border_width = 2
        elif node_type == "MAX":
            bg_color = WHITE
            border_color = PLAYER_O_COLOR
            border_width = 2
        else:
            bg_color = WHITE
            border_color = PLAYER_X_COLOR
            border_width = 2

        pygame.draw.rect(self.screen, bg_color, mini_board_rect, border_radius=3)
        pygame.draw.rect(self.screen, border_color, mini_board_rect, border_width, border_radius=3)

        for i in range(1, 3):
            pygame.draw.line(self.screen, BOARD_LINES,
                             (x + i * MINI_CELL_SIZE, y + 2),
                             (x + i * MINI_CELL_SIZE, y + MINI_BOARD_SIZE - 2), 1)
            pygame.draw.line(self.screen, BOARD_LINES,
                             (x + 2, y + i * MINI_CELL_SIZE),
                             (x + MINI_BOARD_SIZE - 2, y + i * MINI_CELL_SIZE), 1)

        for row in range(3):
            for col in range(3):
                cell_x = x + col * MINI_CELL_SIZE
                cell_y = y + row * MINI_CELL_SIZE

                if board[row][col] == 'X':
                    offset = 3
                    pygame.draw.line(self.screen, PLAYER_X_COLOR,
                                     (cell_x + offset, cell_y + offset),
                                     (cell_x + MINI_CELL_SIZE - offset, cell_y + MINI_CELL_SIZE - offset), 2)
                    pygame.draw.line(self.screen, PLAYER_X_COLOR,
                                     (cell_x + MINI_CELL_SIZE - offset, cell_y + offset),
                                     (cell_x + offset, cell_y + MINI_CELL_SIZE - offset), 2)
                elif board[row][col] == 'O':
                    pygame.draw.circle(self.screen, PLAYER_O_COLOR,
                                       (cell_x + MINI_CELL_SIZE // 2, cell_y + MINI_CELL_SIZE // 2),
                                       MINI_CELL_SIZE // 2 - 3, 2)

        # Dibujar mejor movimiento solo en el nodo raíz
        if is_root and self.agent.decision_tree and self.agent.decision_tree[0].get('best_move'):
            best_move = self.agent.decision_tree[0]['best_move']
            if best_move:
                move_text = font_tiny.render(f"Mejor: ({best_move[0]},{best_move[1]})", True, ACCENT_YELLOW)
                text_x = x + (MINI_BOARD_SIZE - move_text.get_width()) // 2
                self.screen.blit(move_text, (text_x, y - 18))

        if not is_terminal and not is_root:
            type_color = border_color
            type_text = font_tiny.render(node_type, True, type_color)
            text_x = x + (MINI_BOARD_SIZE - type_text.get_width()) // 2
            self.screen.blit(type_text, (text_x, y - 16))

       
        if score is not None and is_terminal:
            score_int = int(score)
            if score_int > 0:
                score_str = f"+{score_int}"
            elif score_int < 0:
                score_str = f"{score_int}"
            else:
                score_str = f"{score_int}"

            score_color = BLACK
            score_text = font_tiny.render(score_str, True, score_color)
            text_x = x + (MINI_BOARD_SIZE - score_text.get_width()) // 2
            text_y = y + MINI_BOARD_SIZE + 4
            self.screen.blit(score_text, (text_x, text_y))

    def draw_decision_tree(self):
        graph_x = LEFT_PANEL_WIDTH + 20
        graph_y = 70
        graph_width = WINDOW_WIDTH - graph_x - 20
        graph_height = WINDOW_HEIGHT - 90

        panel_rect = pygame.Rect(graph_x - 10, graph_y - 10, graph_width + 20, graph_height + 20)
        pygame.draw.rect(self.screen, WHITE, panel_rect, border_radius=15)
        pygame.draw.rect(self.screen, PRIMARY_BLUE, panel_rect, 2, border_radius=15)

        title_shadow = font_subtitle.render("ÁRBOL DE DECISIONES MINIMAX", True, BLACK)
        title = font_subtitle.render("ÁRBOL DE DECISIONES MINIMAX", True, PRIMARY_BLUE)
        title_x = panel_rect.centerx - title.get_width() // 2
        self.screen.blit(title_shadow, (title_x + 1, graph_y - 40))
        self.screen.blit(title, (title_x, graph_y - 41))

        if self.agent.decision_tree:
            tree_start_x = graph_x + 20
            tree_start_y = graph_y + 60
            tree_width = graph_width - 40
            tree_height = graph_height - 100
            self.draw_tree_structure(self.agent.decision_tree[0], tree_start_x, tree_start_y, tree_width, tree_height)
        else:
            no_tree = font_medium.render("Haz clic en el tablero para ver el análisis", True, DARK_GRAY)
            self.screen.blit(no_tree, (graph_x + graph_width // 2 - no_tree.get_width() // 2, graph_y + 150))

        self.draw_legend_left(panel_rect)
        self.draw_stats_right(panel_rect)

    def draw_legend_left(self, panel_rect):
        legend_x = panel_rect.x + 20
        legend_y = panel_rect.y + panel_rect.height - 50

        legend_items = [
            (PLAYER_O_COLOR, "MAX (IA)"),
            (PLAYER_X_COLOR, "MIN (Humano)"),
            ((255, 200, 0), "Terminal"),
            (BLACK, "Actual")
        ]

        start_x = legend_x
        start_y = legend_y
        item_spacing = 130

        for i, (color, text) in enumerate(legend_items):
            x_pos = start_x + i * item_spacing
            box_size = 16
            pygame.draw.rect(self.screen, WHITE, (x_pos, start_y, box_size, box_size), border_radius=3)
            pygame.draw.rect(self.screen, color, (x_pos, start_y, box_size, box_size), 2, border_radius=3)

            text_surface = font_legend.render(text, True, BLACK)
            self.screen.blit(text_surface, (x_pos + box_size + 8, start_y - 2))

    def draw_stats_right(self, panel_rect):
        stats_x = panel_rect.x + panel_rect.width - 350
        stats_y = panel_rect.y + panel_rect.height - 50

        if self.agent.nodes_visited > 0:
            nodes_text = font_legend.render(f"Nodos evaluados: {self.agent.nodes_visited}", True, BLACK)
            self.screen.blit(nodes_text, (stats_x, stats_y))

            depth_text = font_legend.render(f"Profundidad: {self.agent.max_tree_depth}", True, BLACK)
            self.screen.blit(depth_text, (stats_x, stats_y + 25))

            children_text = font_legend.render(f"Hijos por nodo: {self.agent.max_children_display}", True, BLACK)
            self.screen.blit(children_text, (stats_x + 180, stats_y))
        else:
            message = font_legend.render("Haz clic en el tablero para comenzar", True, DARK_GRAY)
            message_x = stats_x - 50
            message_y = stats_y + 10
            self.screen.blit(message, (message_x, message_y))

    def draw_tree_structure(self, node, start_x, start_y, available_width, available_height):
        if not node:
            return
        self.tree_layout.calculate_positions(node, start_x, start_y)
        self.draw_connections(node)
        self.draw_nodes_recursive(node)

    def draw_connections(self, node):
        if not node.get('children'):
            return

        for child in node['children']:
            if 'screen_x' in node and 'screen_y' in node and 'screen_x' in child and 'screen_y' in child:
                start_x = node['screen_x']
                start_y = node['screen_y'] + MINI_BOARD_SIZE // 2
                end_x = child['screen_x']
                end_y = child['screen_y'] - MINI_BOARD_SIZE // 2
                pygame.draw.line(self.screen, SECONDARY_BLUE,
                                 (start_x, start_y),
                                 (end_x, end_y), 1)
            self.draw_connections(child)

    def draw_nodes_recursive(self, node, is_root=True):
        if 'screen_x' not in node or 'screen_y' not in node:
            return

        node_type = "MAX" if node.get('maximizing', True) else "MIN"
        if node.get('terminal', False):
            node_type = "TERM"

        is_current = False
        if self.agent.decision_tree and id(node) == id(self.agent.decision_tree[0]):
            is_current = True

        board_x = int(node['screen_x']) - MINI_BOARD_SIZE // 2
        board_y = int(node['screen_y']) - MINI_BOARD_SIZE // 2

        self.draw_mini_board(node['board'],
                             board_x,
                             board_y,
                             node.get('score'),
                             node_type,
                             is_current=is_current,
                             is_terminal=node.get('terminal', False),
                             is_root=is_root)

        if node.get('children'):
            for child in node['children']:
                self.draw_nodes_recursive(child, is_root=False)

    def draw_simple_button(self, rect, text, color, text_color=WHITE):
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=10)

        text_surface = font_medium.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def draw_new_game_button(self):
        self.draw_simple_button(self.new_game_button, "NUEVO JUEGO", (30, 160, 80), WHITE)

        if self.state.game_over:
            result_y = self.new_game_button.y + 70

            if self.state.winner == 'X':
                result_text = "¡GANASTE!"
                result_color = PLAYER_X_COLOR
            elif self.state.winner == 'O':
                result_text = "GANA LA IA"
                result_color = PLAYER_O_COLOR
            else:
                result_text = "EMPATE"
                result_color = ACCENT_YELLOW

            result_shadow = font_subtitle.render(result_text, True, BLACK)
            result_surface = font_subtitle.render(result_text, True, result_color)
            result_rect = result_surface.get_rect(center=(BOARD_OFFSET_X + BOARD_WIDTH // 2, result_y))

            self.screen.blit(result_shadow, (result_rect.x + 1, result_rect.y + 1))
            self.screen.blit(result_surface, result_rect)

    def draw_close_button(self):
        button_rect = pygame.Rect(WINDOW_WIDTH - 45, 12, 30, 30)
        pygame.draw.rect(self.screen, PLAYER_X_COLOR, button_rect, border_radius=6)
        pygame.draw.rect(self.screen, WHITE, button_rect, 2, border_radius=6)

        x_text = font_small.render("X", True, WHITE)
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

                    close_button = pygame.Rect(WINDOW_WIDTH - 45, 12, 30, 30)
                    if close_button.collidepoint(mouse_pos):
                        running = False

            self.draw_dark_gradient_background()

            pygame.draw.line(self.screen, PRIMARY_BLUE,
                             (LEFT_PANEL_WIDTH + 5, BOARD_OFFSET_Y - 30),
                             (LEFT_PANEL_WIDTH + 5, BOARD_OFFSET_Y + BOARD_HEIGHT + 80), 1)

            self.draw_classic_board()
            self.draw_decision_tree()
            self.draw_new_game_button()
            self.draw_close_button()

            if (not self.state.game_over and self.state.current_player == 'O'):
                thinking_text = font_medium.render("IA pensando...", True, ACCENT_YELLOW)
                text_rect = thinking_text.get_rect(center=(BOARD_OFFSET_X + BOARD_WIDTH // 2,
                                                           BOARD_OFFSET_Y + BOARD_HEIGHT + 20))
                self.screen.blit(thinking_text, text_rect)

                pygame.display.flip()
                pygame.time.wait(300)

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
    
