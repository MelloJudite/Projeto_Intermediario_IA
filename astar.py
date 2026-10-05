import heapq
import math

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


class AStarPathfinder:
    DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(
        self,
        map_array: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int] | None = None,
        wall_influence: float = 10.0,
        buffer_factor: float = 2.0,
        safety_margin: float = 2.0,
        exit_side: str = 'east',
        start_snap_radius: int = 10,
        frontier_band: float = 1.0,
        wall_margin: float | None = None,
    ):
        """Cria um planejador em coordenadas de pixel (linha, coluna).

        Sem goal explícito, planeja em direção à borda indicada por exit_side.
        Se ela ainda não estiver mapeada, retorna o trecho seguro conhecido mais
        próximo dela. Folgas e custos são expressos em pixels do mapa.

        Se o início (posição do robô) cair fora da margem segura, ele é movido
        para a célula segura mais próxima dentro de start_snap_radius pixels.

        safety_margin: folga mínima (px) de paredes E de áreas desconhecidas.
        wall_margin: folga extra (px) só das paredes. Se None, usa safety_margin.
            Use um valor maior que safety_margin para afastar mais o robô das paredes.
            Também sela vãos entre pixels de parede (paredes pontilhadas/diagonais
            por vazamento do laser): vãos menores que ~2*(wall_margin + 0.7) px
            ficam bloqueados.
        """
        if wall_influence < 0 or buffer_factor < 0:
            raise ValueError('Pesos e distâncias não podem ser negativos.')
        if safety_margin < 0:
            raise ValueError('A margem de segurança não pode ser negativa.')
        if wall_margin is not None and wall_margin < 0:
            raise ValueError('wall_margin não pode ser negativo.')
        if exit_side not in ('north', 'east', 'south', 'west'):
            raise ValueError('exit_side deve ser north, east, south ou west.')
        if len(start) != 2:
            raise ValueError('start deve ser (linha, coluna).')

        self.map_array = self.preprocess_map(map_array)
        self.map = self.map_array
        self.start = tuple(int(value) for value in start)
        self.original_start = self.start
        self.wall_influence = float(wall_influence)
        self.buffer_factor = float(buffer_factor)
        self.safety_margin = float(safety_margin)
        self.exit_side = exit_side
        self.start_snap_radius = int(start_snap_radius)
        self.frontier_band = float(frontier_band)
        self.wall_margin = self.safety_margin if wall_margin is None else max(float(wall_margin), self.safety_margin)
        self.GOAL_REACHEABLE = False

        self.wall_mask = self.create_wall_mask()
        self.free_mask = (self.map_array == 255) & ~self.wall_mask
        self.wall_clearance = self.create_wall_clearance()
        padded_free = np.pad(self.free_mask, 1, mode='constant', constant_values=True)
        self.clearance_map = distance_transform_edt(padded_free)[1:-1, 1:-1]
        self.safe_map = self.create_safety_margin()
        self.potential_field = self.create_potential_field()
        self.frontier_mask = self.create_frontier_mask()

        # O robô real quase nunca está exatamente numa célula com folga total:
        # ajusta o início para a célula segura mais próxima.
        if self._inside_map(self.start) and not self.safe_map[self.start]:
            snapped = self._nearest_safe_cell(self.start, self.start_snap_radius)
            if snapped is not None:
                print(f'Início {self.start} fora da margem; ajustado para {snapped}.')
                self.start = snapped

        self.goal_is_exit = goal is None
        if goal is None:
            self.goal = self._border_goal()
        else:
            self.goal = tuple(int(value) for value in goal)

    def _border_goal(self) -> tuple[int, int]:
        rows, columns = self.map_array.shape
        if self.exit_side == 'north':
            return 0, self.start[1]
        if self.exit_side == 'south':
            return rows - 1, self.start[1]
        if self.exit_side == 'west':
            return self.start[0], 0
        return self.start[0], columns - 1

    @staticmethod
    def preprocess_map(map_array: np.ndarray) -> np.ndarray:
        """Normaliza o mapa para parede=0, desconhecido=128 e livre=255."""
        source = np.asarray(map_array)
        if source.ndim != 2:
            raise ValueError('O mapa precisa ser uma imagem em escala de cinza 2D.')

        result = np.zeros(source.shape, dtype=np.uint8)
        result[(source == 128) | (source == 205)] = 128
        result[source >= 240] = 255
        return result

    def create_potential_field(self) -> np.ndarray:
        """Aumenta o custo perto de paredes e de áreas ainda não mapeadas."""
        influence_range = max(self.buffer_factor, 1.0)
        return self.wall_influence * np.exp(-self.clearance_map / influence_range)

    def create_wall_mask(self) -> np.ndarray:
        """Pixels de parede do mapa."""
        return self.map_array == 0

    def create_wall_clearance(self) -> np.ndarray:
        """Distância (px) de cada célula até a parede mais próxima."""
        if not self.wall_mask.any():
            return np.full(self.wall_mask.shape, np.inf)
        return distance_transform_edt(~self.wall_mask)

    def create_safety_margin(self) -> np.ndarray:
        """Bloqueia células perto demais de paredes (borda wall_margin) ou do desconhecido."""
        half_cell = math.sqrt(2) / 2
        return (
            self.free_mask
            & (self.clearance_map >= self.safety_margin + half_cell)
            & (self.wall_clearance >= self.wall_margin + half_cell)
        )

    def create_frontier_mask(self) -> np.ndarray:
        """Células seguras que encostam em área desconhecida (fronteiras de exploração).

        Pixels desconhecidos isolados (riscos de 1 px junto às paredes) são
        removidos com uma abertura morfológica, para não virarem fronteira falsa.
        """
        unknown = (self.map_array == 128).astype(np.uint8)
        solid_unknown = cv2.morphologyEx(unknown, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        if not solid_unknown.any():
            return np.zeros_like(self.safe_map)

        distance_to_unknown = distance_transform_edt(~solid_unknown)
        limit = self.safety_margin + math.sqrt(2) / 2 + self.frontier_band
        return self.safe_map & (distance_to_unknown <= limit)

    @staticmethod
    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        """Distância Manhattan, adequada aos movimentos ortogonais do robô."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _inside_map(self, point: tuple[int, int]) -> bool:
        return 0 <= point[0] < self.map_array.shape[0] and 0 <= point[1] < self.map_array.shape[1]

    def _nearest_safe_cell(self, point: tuple[int, int], radius: int) -> tuple[int, int] | None:
        """Célula segura mais próxima de point dentro de radius pixels."""
        if not self._inside_map(point):
            return None
        if self.safe_map[point]:
            return point

        rows, columns = self.safe_map.shape
        row_min = max(point[0] - radius, 0)
        row_max = min(point[0] + radius + 1, rows)
        col_min = max(point[1] - radius, 0)
        col_max = min(point[1] + radius + 1, columns)

        cells = np.argwhere(self.safe_map[row_min:row_max, col_min:col_max])
        if cells.size == 0:
            return None
        cells = cells + np.array([row_min, col_min])
        distances = np.hypot(cells[:, 0] - point[0], cells[:, 1] - point[1])
        best = int(np.argmin(distances))
        if distances[best] > radius:
            return None
        return int(cells[best, 0]), int(cells[best, 1])

    def _distance_to_exit(self, point: tuple[int, int]) -> int:
        rows, columns = self.map_array.shape
        if self.exit_side == 'north':
            return point[0]
        if self.exit_side == 'south':
            return rows - 1 - point[0]
        if self.exit_side == 'west':
            return point[1]
        return columns - 1 - point[1]

    def _is_exit(self, point: tuple[int, int]) -> bool:
        row, column = point
        rows, columns = self.map_array.shape
        return {
            'north': row == 0,
            'east': column == columns - 1,
            'south': row == rows - 1,
            'west': column == 0,
        }[self.exit_side]

    def find_path(self):
        """Executa A* somente sobre células livres que respeitam a margem."""
        if not self._inside_map(self.start) or not self._inside_map(self.goal):
            print('Início ou destino fora dos limites do mapa.')
            return None, None
        if not self.safe_map[self.start]:
            print('O início não é conhecido como livre com a margem configurada.')
            return None, None

        start_heuristic = self._distance_to_exit(self.start) if self.goal_is_exit else self.heuristic(self.start, self.goal)
        frontier = [(start_heuristic, 0.0, self.start)]
        came_from = {}
        cost_so_far = {self.start: 0.0}
        best_node = self.start
        best_score = (start_heuristic, 0.0)
        best_frontier = None
        best_frontier_score = math.inf

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current_cost != cost_so_far.get(current):
                continue

            current_heuristic = self._distance_to_exit(current) if self.goal_is_exit else self.heuristic(current, self.goal)
            if (current_heuristic, current_cost) < best_score:
                best_node = current
                best_score = (current_heuristic, current_cost)
            if current != self.start and self.frontier_mask[current]:
                frontier_score = current_cost + current_heuristic
                if frontier_score < best_frontier_score:
                    best_frontier = current
                    best_frontier_score = frontier_score
            if (self._is_exit(current) if self.goal_is_exit else current == self.goal):
                self.GOAL_REACHEABLE = True
                return came_from, current

            for row_delta, column_delta in self.DIRECTIONS:
                neighbor = (current[0] + row_delta, current[1] + column_delta)
                if not self._inside_map(neighbor) or not self.safe_map[neighbor]:
                    continue

                step_cost = 1.0 + self.potential_field[neighbor]
                candidate_cost = current_cost + step_cost
                if candidate_cost >= cost_so_far.get(neighbor, math.inf):
                    continue

                came_from[neighbor] = current
                cost_so_far[neighbor] = candidate_cost
                heuristic = self._distance_to_exit(neighbor) if self.goal_is_exit else self.heuristic(neighbor, self.goal)
                priority = candidate_cost + heuristic
                heapq.heappush(frontier, (priority, candidate_cost, neighbor))

        # Borda inalcançável: explora a melhor fronteira (limite entre livre e
        # desconhecido). Ir só ao ponto mais perto da borda prende o robô em becos
        # sem saída, pois o início já pode ser o ponto mais próximo dela.
        if best_frontier is not None:
            print('Borda ainda desconhecida; indo até a melhor fronteira de exploração.')
            return came_from, best_frontier

        if best_node != self.start:
            print('Sem fronteira alcançável; indo ao trecho seguro mais próximo da borda.')
            return came_from, best_node

        self.GOAL_REACHEABLE = False
        print('Nenhum caminho seguro conhecido avança em direção à borda selecionada.')
        return None, None

    def reconstruct_path(self, came_from: dict, current: tuple[int, int]) -> list[tuple[int, int]]:
        """Reconstrói a sequência de células desde o início até o destino parcial."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append((current[0], current[1]))
        path.reverse()
        return path

    def know_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Corta o caminho no primeiro pixel desconhecido ou fora da margem."""
        safe_path = []
        for point in path:
            if not self._inside_map(point) or not self.safe_map[point]:
                break
            safe_path.append(point)
        return safe_path

    def _line_is_safe(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """Verifica as células tocadas por uma linha, incluindo cruzamentos diagonais."""
        row, column = start
        row_delta = end[0] - row
        column_delta = end[1] - column
        row_steps = abs(row_delta)
        column_steps = abs(column_delta)
        row_direction = 1 if row_delta > 0 else -1
        column_direction = 1 if column_delta > 0 else -1
        row_progress = 0
        column_progress = 0
        cells = [(row, column)]

        while row_progress < row_steps or column_progress < column_steps:
            decision = (1 + 2 * column_progress) * row_steps - (1 + 2 * row_progress) * column_steps
            if decision == 0:
                cells.extend(((row, column + column_direction), (row + row_direction, column)))
                row += row_direction
                column += column_direction
                row_progress += 1
                column_progress += 1
            elif decision < 0:
                column += column_direction
                column_progress += 1
            else:
                row += row_direction
                row_progress += 1
            cells.append((row, column))

        return all(self._inside_map(point) and self.safe_map[point] for point in cells)

    def simplify_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Usa o waypoint mais distante visível sem sair da margem segura."""
        if len(path) <= 2:
            return path.copy()

        simplified = [path[0]]
        index = 0
        while index < len(path) - 1:
            candidate = len(path) - 1
            while candidate > index + 1 and not self._line_is_safe(path[index], path[candidate]):
                candidate -= 1
            simplified.append(path[candidate])
            index = candidate
        return simplified

    def plot_path(self, path: list[tuple[int, int]], simplified_path: list[tuple[int, int]]):
        """Exibe apenas o trecho seguro conhecido, além do destino solicitado."""
        import matplotlib.pyplot as plt  # só é importado se alguém pedir o gráfico

        plt.figure(figsize=(10, 7))
        plt.imshow(self.map_array, cmap='gray', vmin=0, vmax=255, origin='upper')
        plt.scatter(self.start[1], self.start[0], color='green', s=80, label='Início')
        plt.scatter(self.goal[1], self.goal[0], color='blue', s=80, label='Destino')

        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_y, path_x, color='magenta', linewidth=1, label='Trecho conhecido seguro')
        if simplified_path:
            way_x, way_y = zip(*simplified_path)
            plt.plot(way_y, way_x, color='red', linewidth=2, linestyle='--', label='Pontos de navegação')

        plt.legend()
        plt.axis('equal')
        plt.show()

    def run(self, show_path: bool = False) -> list[tuple[int, int]]:
        """Retorna os waypoints seguros até a borda.

        A lista pode vir VAZIA (sem caminho seguro) ou com 1 ponto apenas.
        Use next_waypoint() em vez de indexar diretamente.
        """
        came_from, final_node = self.find_path()
        if final_node is None:
            return []

        planned_path = self.reconstruct_path(came_from, final_node)
        safe_path = self.know_path(planned_path)
        if not safe_path:
            print('Nenhum trecho conhecido com folga suficiente para avançar.')
            return []

        if len(safe_path) == 1 and self.start != self.goal:
            print('Sem avanço seguro conhecido; atualize o mapa antes de continuar.')
            return []

        waypoints = self.simplify_path(safe_path)
        if show_path:
            self.plot_path(safe_path, waypoints)
        return waypoints

    @staticmethod
    def next_waypoint(waypoints: list[tuple[int, int]]) -> tuple[int, int] | None:
        """Próximo alvo do robô (pula o ponto atual). None se não há para onde ir."""
        if len(waypoints) >= 2:
            return waypoints[1]
        if len(waypoints) == 1:
            return waypoints[0]
        return None


def prep_map(map_path: str) -> np.ndarray:
    """Carrega um PGM sem filtrar paredes finas e mantém a orientação do template."""
    map_array = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if map_array is None:
        raise FileNotFoundError(f'Não foi possível carregar o mapa: {map_path}')

    normalized = AStarPathfinder.preprocess_map(map_array)
    return np.flipud(normalized).copy()


def main():
    start = (60, 20)
    for snapshot in range(1, 6):
        map_array = prep_map(f'map{snapshot}.pgm')
        astar = AStarPathfinder(
            map_array,
            start,
            wall_influence=10.0,
            buffer_factor=3.0,
            safety_margin=2.0,
            wall_margin=4.0,
            exit_side='east',
        )
        waypoints = astar.run()
        print(f'map{snapshot}: início={start}, waypoints={waypoints}')

        target = AStarPathfinder.next_waypoint(waypoints)
        if target is None:
            print('Sem alvo seguro; robô deve parar/girar e atualizar o mapa.')
            continue
        start = waypoints[-1]


if __name__ == '__main__':
    main()