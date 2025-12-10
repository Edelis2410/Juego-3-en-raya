import pygame
import sys
import math
import time

pygame.init()

# --- CONSTANTES Y COLORES ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PRIMARY_BLUE = (25, 130, 196)
ACCENT_YELLOW = (255, 200, 50)
PLAYER_X_COLOR = (220, 60, 60)   # Rojo (MAX)
PLAYER_O_COLOR = (60, 130, 220)  # Azul (MIN)
CURRENT_BOARD_COLOR = (255, 165, 0) 
MEDIUM_GRAY = (200, 210, 220)
DARK_GRAY = (40, 50, 65)
BOARD_LINES = (0, 0, 0)
BACKGROUND_DARK = (25, 15, 55)
CLOSE_BUTTON_RED = (200, 30, 30)

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 850

BOARD_SIZE = 3
CELL_SIZE = 90
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE

LEFT_PANEL_WIDTH = 380
BOARD_OFFSET_X = (LEFT_PANEL_WIDTH - BOARD_WIDTH) // 2
BOARD_OFFSET_Y = (WINDOW_HEIGHT - BOARD_HEIGHT) // 2

# Tamaño nodos visuales
MINI_CELL_SIZE = 12
MINI_BOARD_SIZE = MINI_CELL_SIZE * 3

# Fuentes
font_title = pygame.font.SysFont('Arial Black', 38, bold=True)
font_subtitle = pygame.font.SysFont('Arial', 24, bold=True)
font_medium = pygame.font.SysFont('Arial', 20)
font_small = pygame.font.SysFont('Arial', 16)
font_tiny = pygame.font.SysFont('Arial', 10)
font_legend = pygame.font.SysFont('Arial', 12)

# --- CLASE ESTADO DEL JUEGO ---
class GameState:
    def __init__(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        self.moves_made = 0
        self.ia_thinking = False

    def reset(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        self.moves_made = 0
        self.ia_thinking = False

    def make_move(self, row, col, player):
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            self.moves_made += 1
            return True
        return False

    def copy(self):
        new_state = GameState()
        new_state.board = [row[:] for row in self.board]
        new_state.current_player = self.current_player
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        new_state.moves_made = self.moves_made
        new_state.ia_thinking = self.ia_thinking
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

# --- CLASE AGENTE MINIMAX  ---
class MinimaxAgent:
    def __init__(self):
        self.nodes_visited = 0
        self.best_move = None
        self.decision_tree = []
        self.MAX_GAME_DEPTH = 9 
        self.max_vis_depth = 3      

    def reset(self):
        self.nodes_visited = 0
        self.decision_tree = []
        self.best_move = None

    def evaluate(self, state):
        winner = state.check_winner()
        if winner == 'X': return 1
        elif winner == 'O': return -1
        elif winner == 'Tie': return 0
        return None

    def minimax(self, state, depth, maximizing_player, node):
        self.nodes_visited += 1

        node['board'] = [row[:] for row in state.board]
        node['depth'] = depth
        node['maximizing'] = maximizing_player
        node['terminal'] = False

        if state.is_terminal():
            score = self.evaluate(state)
            node['score'] = score
            node['terminal'] = True
            return score

        if depth >= self.MAX_GAME_DEPTH:
            node['score'] = 0 
            return 0

        empty_cells = state.get_empty_cells()
        
        center_first = []
        corners = []
        edges = []
        for (row, col) in empty_cells:
            if (row, col) == (1, 1): center_first.append((row, col))
            elif (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]: corners.append((row, col))
            else: edges.append((row, col))
        ordered_moves = center_first + corners + edges
        
        if depth == 0:
            max_vis_children = 9 
        elif depth == 2:
            max_vis_children = 1 
        else:
            max_vis_children = 2 

        if maximizing_player:
            best_val = -math.inf
            
            for (row, col) in ordered_moves:
                new_state = state.copy()
                new_state.board[row][col] = 'X'

                child_node = {'children': [], 'move': (row, col)}
                
                can_display_depth = (depth < self.max_vis_depth)
                can_display_width = (len(node['children']) < max_vis_children) 

                if can_display_depth and can_display_width:
                    node['children'].append(child_node)

                val = self.minimax(new_state, depth + 1, False, child_node)
                
                if val > best_val:
                    best_val = val
                    if depth == 0:
                        self.best_move = (row, col)
                elif val == best_val and depth == 0:
                    best_val = val
                
            node['score'] = best_val
            return best_val
        else:
            best_val = math.inf
            
            for (row, col) in ordered_moves:
                new_state = state.copy()
                new_state.board[row][col] = 'O'

                child_node = {'children': [], 'move': (row, col)}
                
                can_display_depth = (depth < self.max_vis_depth)
                can_display_width = (len(node['children']) < max_vis_children)

                if can_display_depth and can_display_width:
                    node['children'].append(child_node)

                val = self.minimax(new_state, depth + 1, True, child_node)
                
                if val < best_val:
                    best_val = val
                    if depth == 0:
                        self.best_move = (row, col)
                elif val == best_val and depth == 0:
                    best_val = val
                
            node['score'] = best_val
            return best_val

    def get_best_move(self, state):
        self.nodes_visited = 0
        self.best_move = None
        self.decision_tree = [] 

        maximizing_root = (state.current_player == 'X')
        
        root = {
            'board': [row[:] for row in state.board],
            'depth': 0,
            'maximizing': maximizing_root,
            'children': [],
            'score': None
        }
        self.decision_tree.append(root)

        self.minimax(state, 0, maximizing_root, root)

        if not self.best_move and state.get_empty_cells():
            self.best_move = state.get_empty_cells()[0]

        return self.best_move

# --- CLASE LAYOUT DEL ÁRBOL  ---
class TreeLayout:
    def __init__(self, vertical_spacing=130, horizontal_padding=25):
        self.vertical_spacing = vertical_spacing
        self.horizontal_padding = horizontal_padding

    def calculate_positions(self, node, start_x, start_y, width):
        if not node: return
        
        node['screen_x'] = start_x + width / 2
        node['screen_y'] = start_y + node['depth'] * self.vertical_spacing

        if not node.get('children'): return

        children = node['children']
        num_children = len(children)
        if num_children == 0: return

        min_unit_width = MINI_BOARD_SIZE + self.horizontal_padding
        total_required_width = num_children * min_unit_width - self.horizontal_padding 

        if width > total_required_width:
            child_width = (width - (num_children - 1) * self.horizontal_padding) / num_children
        else:
            child_width = MINI_BOARD_SIZE 
        child_width = max(MINI_BOARD_SIZE, child_width)
        
        children_total_occupied_width = num_children * child_width + (num_children - 1) * self.horizontal_padding
        current_x = start_x + max(0, (width - children_total_occupied_width) / 2)

        for i, child in enumerate(children):
            child_start_x = current_x
            self.calculate_positions(child, child_start_x, start_y, child_width)
            current_x += child_width + self.horizontal_padding

# --- CLASE INTERFAZ GRÁFICA ---
class GameGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tres en Raya - Árbol Minimax CORREGIDO")
        self.clock = pygame.time.Clock()
        self.state = GameState()
        self.agent = MinimaxAgent()
        
        self.tree_layout = TreeLayout(vertical_spacing=130, horizontal_padding=25)

        button_width = 180
        button_x = BOARD_OFFSET_X + (BOARD_WIDTH - button_width) // 2
        self.new_game_button = pygame.Rect(button_x, BOARD_OFFSET_Y + BOARD_HEIGHT + 30, button_width, 45)

    def draw_dark_gradient_background(self):
        self.screen.fill(BACKGROUND_DARK)
        
    def draw_classic_board(self):
        board_rect = pygame.Rect(BOARD_OFFSET_X - 10, BOARD_OFFSET_Y - 10, BOARD_WIDTH + 20, BOARD_HEIGHT + 20)
        pygame.draw.rect(self.screen, WHITE, board_rect, border_radius=15)
        pygame.draw.rect(self.screen, BLACK, board_rect, 3, border_radius=15)

        line_width = 6
        for i in range(1, BOARD_SIZE):
            x_pos = BOARD_OFFSET_X + i * CELL_SIZE
            pygame.draw.line(self.screen, BOARD_LINES, (x_pos, BOARD_OFFSET_Y), (x_pos, BOARD_OFFSET_Y + BOARD_HEIGHT), line_width)
            y_pos = BOARD_OFFSET_Y + i * CELL_SIZE
            pygame.draw.line(self.screen, BOARD_LINES, (BOARD_OFFSET_X, y_pos), (BOARD_OFFSET_X + BOARD_WIDTH, y_pos), line_width)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.state.board[row][col] != ' ':
                    cell_x = BOARD_OFFSET_X + col * CELL_SIZE + (CELL_SIZE // 2)
                    cell_y = BOARD_OFFSET_Y + row * CELL_SIZE + (CELL_SIZE // 2)

                    if self.state.board[row][col] == 'X':
                        pygame.draw.line(self.screen, PLAYER_X_COLOR, (cell_x - 25, cell_y - 25), (cell_x + 25, cell_y + 25), 10)
                        pygame.draw.line(self.screen, PLAYER_X_COLOR, (cell_x + 25, cell_y - 25), (cell_x - 25, cell_y + 25), 10)
                    elif self.state.board[row][col] == 'O':
                        pygame.draw.circle(self.screen, PLAYER_O_COLOR, (cell_x, cell_y), 30, 8)

        title = font_title.render("3 EN RAYA", True, ACCENT_YELLOW)
        self.screen.blit(title, (BOARD_OFFSET_X + (BOARD_WIDTH - title.get_width()) // 2, BOARD_OFFSET_Y - 60))

        if self.state.ia_thinking:
            msg = font_subtitle.render("IA PENSANDO...", True, ACCENT_YELLOW)
            msg_rect = msg.get_rect(center=(BOARD_OFFSET_X + BOARD_WIDTH // 2, BOARD_OFFSET_Y + BOARD_HEIGHT // 2))
            
            s = pygame.Surface((BOARD_WIDTH, 50), pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, (BOARD_OFFSET_X, msg_rect.y - 10))
            
            self.screen.blit(msg, msg_rect)

    def draw_mini_board(self, board, x, y, score=None, node_type="MAX", terminal=False, is_current=False):
        mini_board_rect = pygame.Rect(x, y, MINI_BOARD_SIZE, MINI_BOARD_SIZE)
        
        if node_type == "MAX":
            border_color = PLAYER_X_COLOR
            label = "MAX"
        elif node_type == "MIN":
            border_color = PLAYER_O_COLOR
            label = "MIN"
        else:
            border_color = DARK_GRAY
            label = ""

        if is_current:
            border_color = CURRENT_BOARD_COLOR
            label = "ACTUAL"

        pygame.draw.rect(self.screen, WHITE, mini_board_rect)
        pygame.draw.rect(self.screen, border_color, mini_board_rect, 1)

        for i in range(1, 3):
            pygame.draw.line(self.screen, BLACK, (x + i * MINI_CELL_SIZE, y), (x + i * MINI_CELL_SIZE, y + MINI_BOARD_SIZE), 1)
            pygame.draw.line(self.screen, BLACK, (x, y + i * MINI_CELL_SIZE), (x + MINI_BOARD_SIZE, y + i * MINI_CELL_SIZE), 1)

        for row in range(3):
            for col in range(3):
                cx = x + col * MINI_CELL_SIZE
                cy = y + row * MINI_CELL_SIZE
                if board[row][col] == 'X':
                    pygame.draw.line(self.screen, PLAYER_X_COLOR, (cx+3, cy+3), (cx+MINI_CELL_SIZE-3, cy+MINI_CELL_SIZE-3), 2)
                    pygame.draw.line(self.screen, PLAYER_X_COLOR, (cx+MINI_CELL_SIZE-3, cy+3), (cx+3, cy+MINI_CELL_SIZE-3), 2)
                elif board[row][col] == 'O':
                    pygame.draw.circle(self.screen, PLAYER_O_COLOR, (cx + MINI_CELL_SIZE//2, cy + MINI_CELL_SIZE//2), MINI_CELL_SIZE//2 - 2, 2)

        if not terminal:
            lbl_surf = font_tiny.render(label, True, border_color)
            self.screen.blit(lbl_surf, (x + (MINI_BOARD_SIZE - lbl_surf.get_width())//2, y - 12))

        if terminal and score is not None:
            if score > 0: score_str = "+1"
            elif score < 0: score_str = "-1"
            else: score_str = "0"
            
            color = BLACK
            if score > 0: color = PLAYER_X_COLOR
            elif score < 0: color = PLAYER_O_COLOR
            
            sc_surf = font_medium.render(score_str, True, color)
            self.screen.blit(sc_surf, (x + (MINI_BOARD_SIZE - sc_surf.get_width())//2, y + MINI_BOARD_SIZE + 2))

    def draw_decision_tree(self):
        graph_x = LEFT_PANEL_WIDTH + 20
        graph_y = 50
        graph_width = WINDOW_WIDTH - graph_x - 20
        graph_height = WINDOW_HEIGHT - 70

        panel_rect = pygame.Rect(graph_x, graph_y, graph_width, graph_height)
        pygame.draw.rect(self.screen, WHITE, panel_rect, border_radius=15)
        pygame.draw.rect(self.screen, PRIMARY_BLUE, panel_rect, 2, border_radius=15)

        title = font_subtitle.render("ÁRBOL MINIMAX", True, PRIMARY_BLUE)
        self.screen.blit(title, (graph_x + (graph_width - title.get_width())//2, graph_y + 15))

        if self.state.moves_made > 0 and self.agent.decision_tree:
            root = self.agent.decision_tree[0]
            tree_area_y = graph_y + 60
            self.tree_layout.calculate_positions(root, graph_x + 20, tree_area_y, graph_width - 40)
            self.draw_tree_connections(root)
            self.draw_tree_nodes(root)
            
            # LEYENDA 
            legend_y = graph_y + graph_height - 55
            legend_bg = pygame.Rect(graph_x + 10, legend_y, 380, 28)
            pygame.draw.rect(self.screen, WHITE, legend_bg, border_radius=8)
            pygame.draw.rect(self.screen, BLACK, legend_bg, 1, border_radius=8)
            

            x_offset = graph_x + 15
            
            
            # Naranja = Actual
            pygame.draw.rect(self.screen, CURRENT_BOARD_COLOR, (x_offset, legend_y + 6, 12, 12), 2)
            actual_text = font_legend.render("Actual", True, BLACK)
            self.screen.blit(actual_text, (x_offset + 18, legend_y + 6))
            x_offset += 90
            
            # Rojo = MAX
            pygame.draw.rect(self.screen, PLAYER_X_COLOR, (x_offset, legend_y + 6, 12, 12), 2)
            max_text = font_legend.render("MAX(X)", True, BLACK)
            self.screen.blit(max_text, (x_offset + 18, legend_y + 6))
            x_offset += 95
            
            # Azul = MIN
            pygame.draw.rect(self.screen, PLAYER_O_COLOR, (x_offset, legend_y + 6, 12, 12), 2)
            min_text = font_legend.render("MIN(O)", True, BLACK)
            self.screen.blit(min_text, (x_offset + 18, legend_y + 6))
            
        else:
            msg = font_medium.render("Realiza el primer movimiento para ver el árbol...", True, DARK_GRAY)
            self.screen.blit(msg, (graph_x + (graph_width - msg.get_width())//2, graph_y + 200))

    def draw_tree_connections(self, node):
        if not node or not node.get('children'): return
        start_x, start_y = node['screen_x'], node['screen_y'] + MINI_BOARD_SIZE
        for child in node['children']:
            end_x, end_y = child['screen_x'], child['screen_y']
            pygame.draw.line(self.screen, MEDIUM_GRAY, (start_x, start_y), (end_x, end_y), 1)
            self.draw_tree_connections(child)

    def draw_tree_nodes(self, node):
        if not node: return
        is_max = node['maximizing']
        node_type = "MAX" if is_max else "MIN"
        bx = int(node['screen_x']) - MINI_BOARD_SIZE // 2
        by = int(node['screen_y'])
        
        is_current_board = (node['depth'] == 0)
        score_to_display = node.get('score') if node.get('terminal') else None
        
        self.draw_mini_board(node['board'], bx, by, score_to_display, node_type, terminal=node.get('terminal', False), is_current=is_current_board)
        if node.get('children'):
            for child in node['children']:
                self.draw_tree_nodes(child)

    def draw_new_game_button(self):
        color = (30, 160, 80)
        pygame.draw.rect(self.screen, color, self.new_game_button, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, self.new_game_button, 2, border_radius=10)
        text = font_medium.render("NUEVO JUEGO", True, WHITE)
        self.screen.blit(text, (self.new_game_button.centerx - text.get_width()//2, self.new_game_button.centery - text.get_height()//2))

        if self.state.game_over:
            if self.state.winner == 'X':
                res = "¡GANASTE!"
                col = PLAYER_X_COLOR
            elif self.state.winner == 'O':
                res = "GANA LA IA ✓"
                col = PLAYER_O_COLOR
            else:
                res = "EMPATE"
                col = ACCENT_YELLOW
            res_surf = font_subtitle.render(res, True, col)
            self.screen.blit(res_surf, (BOARD_OFFSET_X + (BOARD_WIDTH - res_surf.get_width())//2, self.new_game_button.y + 60))

    def draw_close_button(self):
        button_rect = pygame.Rect(WINDOW_WIDTH - 50, 45, 35, 35)
        pygame.draw.rect(self.screen, CLOSE_BUTTON_RED, button_rect, border_radius=8)
        pygame.draw.rect(self.screen, WHITE, button_rect, 3, border_radius=8)
        x_text = font_medium.render("X", True, WHITE)
        x_rect = x_text.get_rect(center=button_rect.center)
        self.screen.blit(x_text, x_rect)

    def get_cell_from_pos(self, pos):
        x, y = pos
        if BOARD_OFFSET_X <= x <= BOARD_OFFSET_X + BOARD_WIDTH and BOARD_OFFSET_Y <= y <= BOARD_OFFSET_Y + BOARD_HEIGHT:
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
                    close_button = pygame.Rect(WINDOW_WIDTH - 50, 25, 35, 35)
                    if close_button.collidepoint(mouse_pos):
                        running = False

                    if not self.state.ia_thinking and not self.state.game_over and self.state.current_player == 'X':
                        cell = self.get_cell_from_pos(mouse_pos)
                        if cell and self.state.make_move(*cell, 'X'):
                            self.state.winner = self.state.check_winner()
                            if self.state.winner:
                                self.state.game_over = True
                            else:
                                self.state.current_player = 'O'
                                self.state.ia_thinking = True
                            
                            self.agent.get_best_move(self.state)
                                 
                    if self.new_game_button.collidepoint(mouse_pos):
                        self.state.reset()
                        self.agent.reset()

            self.draw_dark_gradient_background()
            self.draw_classic_board()
            self.draw_decision_tree()
            self.draw_new_game_button()
            self.draw_close_button()

            if not self.state.game_over and self.state.current_player == 'O' and self.state.ia_thinking:
                pygame.display.flip() 
                time.sleep(1) 

                best = self.agent.best_move 
                
                if not best:
                    best = self.agent.get_best_move(self.state) 

                if best:
                    self.state.make_move(best[0], best[1], 'O')
                    self.state.winner = self.state.check_winner()
                    if self.state.winner:
                        self.state.game_over = True
                    else:
                        self.state.current_player = 'X'
                    
                    if not self.state.game_over:
                        self.agent.get_best_move(self.state)

                self.state.ia_thinking = False

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = GameGUI()
    game.run()
