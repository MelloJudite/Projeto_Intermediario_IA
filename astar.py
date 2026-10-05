import heapq
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

LIVRE = 255        
DESCONHECIDO = 128 
PAREDE = 0      

class AStarPathfinder:
    def __init__(self, map_array, start, goal, wall_influence=5.0, buffer_factor=2.0):
        self.start = start
        self.goal = goal
        self.wall_influence = wall_influence
        self.buffer_factor = buffer_factor
        self.GOAL_REACHEABLE = False

        self.min_clearance = 4.0

        self.map = map_array.copy()
        self.map_array = self.preprocess_map(map_array)
        self.potential_field = self.create_potential_field()

    # ---------- preparação ----------

    def preprocess_map(self, map_array):
        processed = map_array.copy()
        processed[(processed != LIVRE) & (processed != DESCONHECIDO)] = PAREDE
        return processed

    def create_potential_field(self):
       
        paredes = (self.map_array == PAREDE)
        self.dist_to_wall = distance_transform_edt(~paredes)
        campo = self.wall_influence * np.exp(-self.dist_to_wall / self.buffer_factor)
        campo[paredes] = 0.0
        return campo

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])


    def find_path(self):
        """Devolve (came_from, nó_final), ou (None, None) se não achar caminho."""
        start, goal = self.start, self.goal
        linhas, colunas = self.map_array.shape

        def dentro(r, c):
            return 0 <= r < linhas and 0 <= c < colunas

        def sem_parede(r, c):
            return self.map_array[r, c] != PAREDE

        if not dentro(*goal) or not sem_parede(*goal):
            goal = self._nearest_free(goal)
            if goal is None:
                print("Caminho não encontrado")
                return None, None
            self.goal = goal

        fila = [(self.heuristic(start, goal), 0.0, start)]  # (f, g, célula)
        came_from = {start: None}
        g_score = {start: 0.0}
        visitados = set()

        raiz2 = math.sqrt(2)
        vizinhos = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                    (-1, -1, raiz2), (-1, 1, raiz2), (1, -1, raiz2), (1, 1, raiz2)]

        while fila:
            _, g, atual = heapq.heappop(fila)
            if atual in visitados:
                continue
            visitados.add(atual)

            if atual == goal:
                self.GOAL_REACHEABLE = (self.map[goal[0], goal[1]] == LIVRE)
                return came_from, atual

            r, c = atual
            for dr, dc, passo in vizinhos:
                nr, nc = r + dr, c + dc
                if not dentro(nr, nc) or not sem_parede(nr, nc):
                    continue

                if dr != 0 and dc != 0:
                    if not sem_parede(r + dr, c) or not sem_parede(r, c + dc):
                        continue

                custo = passo + self.potential_field[nr, nc]
                if self.map_array[nr, nc] == DESCONHECIDO:
                    custo += 0.1  # preferência leve pelo que já é conhecido

                novo_g = g + custo
                if novo_g < g_score.get((nr, nc), math.inf):
                    g_score[(nr, nc)] = novo_g
                    came_from[(nr, nc)] = atual
                    f = novo_g + self.heuristic((nr, nc), goal)
                    heapq.heappush(fila, (f, novo_g, (nr, nc)))

        print("Caminho não encontrado")
        return None, None

    def _nearest_free(self, cell):
        r0, c0 = int(round(cell[0])), int(round(cell[1]))
        livres = np.argwhere(self.map_array != PAREDE)
        if len(livres) == 0:
            return None

        melhor, melhor_d = None, math.inf
        for r, c in livres:
            d = (r - r0) ** 2 + (c - c0) ** 2
            if d < melhor_d:
                melhor_d, melhor = d, (int(r), int(c))
        return melhor

    def reconstruct_path(self, came_from, current):
        path = [current]
        while came_from.get(path[-1]) is not None:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def know_path(self, path):
       
        if not path:
            return path

        conhecido = [path[0]]
        for cell in path[1:]:
            if self.map[cell[0], cell[1]] != LIVRE:
                break
            conhecido.append(cell)
        return conhecido


    def simplify_path(self, path):
      
        if not path or len(path) < 3:
            return path

        reduzido = [path[0]]
        for i in range(1, len(path) - 1):
            dir_antes = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            dir_depois = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if dir_antes != dir_depois:
                reduzido.append(path[i])
        reduzido.append(path[-1])

        suave = [reduzido[0]]
        ancora = 0
        while ancora < len(reduzido) - 1:
            proximo = ancora + 1
            for cand in range(len(reduzido) - 1, ancora, -1):
                if self._segmento_seguro(reduzido[ancora], reduzido[cand]):
                    proximo = cand
                    break
            suave.append(reduzido[proximo])
            ancora = proximo

        return self._chaikin_seguro(suave, iteracoes=3)

    def _chaikin_seguro(self, pontos, iteracoes=3):
      
        pts = [tuple(map(float, p)) for p in pontos]

        for _ in range(iteracoes):
            if len(pts) < 3:
                break

            novo = [pts[0]]
            for i in range(1, len(pts) - 1):
                a, v, b = novo[-1], pts[i], pts[i + 1]
                q1 = (0.75 * v[0] + 0.25 * a[0], 0.75 * v[1] + 0.25 * a[1])
                q2 = (0.75 * v[0] + 0.25 * b[0], 0.75 * v[1] + 0.25 * b[1])

                if self._segmento_seguro(a, q1) and self._segmento_seguro(q1, q2):
                    novo.extend([q1, q2])
                else:
                    novo.append(v)
            novo.append(pts[-1])
            pts = novo

        return pts

    def _segmento_seguro(self, a, b):
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(dist * 3) + 1)

        for t in np.linspace(0, 1, n):
            r = int(round(a[0] + (b[0] - a[0]) * t))
            c = int(round(a[1] + (b[1] - a[1]) * t))
            if self.map_array[r, c] == PAREDE:
                return False
            if self.dist_to_wall[r, c] < self.min_clearance:
                return False
        return True

    def path_to_xy(self, path):
        return [(round(float(c), 2), round(float(r), 2)) for r, c in path]

    def plot_path(self, path, simplified_path, save_path=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.map, cmap='gray')
        plt.scatter(self.start[1], self.start[0], color='green', s=100, label='Início')
        plt.scatter(self.goal[1], self.goal[0], color='blue', s=100, label='Objetivo')

        if path:
            linhas, colunas = zip(*path)
            plt.plot(colunas, linhas, color='magenta', linewidth=1, label='Caminho completo')
            s_linhas, s_colunas = zip(*simplified_path)
            plt.plot(s_colunas, s_linhas, color='red', linewidth=2,
                     linestyle='--', label='Caminho simplificado')
        else:
            plt.title("Caminho não encontrado")

        plt.legend()
        plt.axis('equal')
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        plt.show()

    def run(self, show_path=True, save_path=None):
        
        print("Procurando caminho...")
        came_from, no_final = self.find_path()

        if not no_final:
            print("Nenhum caminho encontrado.")
            self.caminho_xy = None
            return None

        path = self.reconstruct_path(came_from, no_final)

        path = self.know_path(path)

        print("Simplificando caminho...")
        simplificado = self.simplify_path(path)

        self.caminho_xy = self.path_to_xy(simplificado)
        print(f"Caminho com {len(self.caminho_xy)} pontos (x, y):")
        print(self.caminho_xy)

        if show_path:
            self.plot_path(path, simplificado, save_path=save_path)

        return simplificado


def prep_map(map_path):
    """Carrega o .pgm e deixa só 0 (parede), 128 (desconhecido) e 255 (livre)."""
    mapa = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    mapa[mapa == 205] = 128
    mapa[mapa == 254] = 255
    mapa[(mapa >= 60) & (mapa != 128) & (mapa != 255)] = 0
    mapa = mapa.astype(np.uint8)

    mapa = cv2.morphologyEx(mapa, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mapa = np.flipud(mapa)

    return np.pad(mapa, ((0, 200), (0, 200)), 'constant', constant_values=128)


def main():
    for i in range(1, 6):
        print(f"\n===== map{i}.pgm =====")
        mapa = prep_map(f'map{i}.pgm')
        astar = AStarPathfinder(mapa, (60, 20), (60, 120),
                                wall_influence=10.0, buffer_factor=3.0)
        caminho = astar.run(save_path=f'caminho_map{i}.png')
        if caminho:
            print(f"Objetivo alcançável (região conhecida): {astar.GOAL_REACHEABLE}")


if __name__ == '__main__':
    main()