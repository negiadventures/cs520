import time
from heapq import heappush, heappop

import numpy as np


class PriorityQueue:

    def __init__(self, iterable=[]):
        self.heap = []
        for value in iterable:
            heappush(self.heap, (0, value))

    def add(self, value, priority=0):
        heappush(self.heap, (priority, value))

    def pop(self):
        priority, value = heappop(self.heap)
        return value

    def __len__(self):
        return len(self.heap)


def get_heuristic(h_fun, dim):
    def calc_h(cell):
        (i, j) = cell
        # MANHATTAN
        return abs(dim - i) + abs(dim - j)

    return calc_h


def calc_cx(nb):
    count = 0
    for cell in nb:
        if grid[cell] == 1:
            count += 1
    return count


def calc_b(nb):
    li = []
    for cell in nb:
        if cell in blocked and blocked[cell] == 1:
            li.append(cell)
    return li


def calc_e(nb):
    li = []
    for cell in nb:
        if cell in blocked and blocked[cell] == 0:
            li.append(cell)
    return li


def calc_h(n, b, e):
    return list(set(n) - set(b) - set(e))


def update_info(child):
    b[child] = calc_b(n[child])
    e[child] = calc_e(n[child])
    h[child] = calc_h(n[child], b[child], e[child])


def update_inference(visited):
    for cell in visited:
        cx = c[cell]
        nx = len(n[cell])
        bx = len(b[cell])
        ex = len(e[cell])
        if cx == bx != 0:
            for h1 in h[cell]:
                blocked[h1] = 0
                if h1 in n:
                    for n1 in n[h1]:
                        update_info(n1)
            e[cell].extend(h[cell])
            ex = len(e[cell])
            h[cell] = []
        if nx - cx == ex:
            for h1 in h[cell]:
                blocked[h1] = 1
                if h1 in n:
                    for n1 in n[h1]:
                        update_info(n1)
            b[cell].extend(h[cell])
            h[cell] = []


def a_star_search(start, neighbors, heuristic, grid):
    dim = len(grid[0])
    visited = set()
    parent = dict()
    distance = {start: 0}
    fringe = PriorityQueue()
    fringe.add(start)
    while fringe:
        cell = fringe.pop()
        if grid[cell] != 1:
            blocked[cell] = 0
        else:
            blocked[cell] = 1
            pass
        if cell in visited:
            continue
        if cell == (dim - 1, dim - 1):
            return reconstruct_path(parent, start, cell)
        visited.add(cell)
        nb, fx = neighbors(cell)
        nb = list(nb)
        n[cell] = nb
        c[cell] = calc_cx(nb)
        b[cell] = calc_b(nb)
        e[cell] = calc_e(nb)
        h[cell] = calc_h(n[cell], b[cell], e[cell])
        hx = len(h[cell])
        for child in list(nb):
            if child not in visited:
                blocked[child] = -1
        for child in fx:
            fringe.add(child, priority=distance[cell] + 1 + heuristic(child))
            if child not in distance or distance[cell] + 1 < distance[child]:
                distance[child] = distance[cell] + 1
                parent[child] = cell
            if child in visited and len(h[child]) != 0:
                update_info(child)
        # print('*** ITERATION ***')
        # print('current:', cell)
        # print('visited:', visited)
        # print('blocked:', blocked)
        # print('n: ', n)
        # print('c: ', c)
        # print('b: ', b)
        # print('e: ', e)
        # print('h: ', h)
        update_inference(visited)
        # print('*** After updated Inference ***')
        # print('current:', cell)
        # print('visited:', visited)
        # print('blocked:', blocked)
        # print('n: ', n)
        # print('c: ', c)
        # print('b: ', b)
        # print('e: ', e)
        # print('h: ', h)
    return None


def reconstruct_path(parent, start, end):
    path = [end]
    while end != start:
        end = parent[end]
        path.append(end)
    return list(reversed(path))


def get_neighbors(grid, dim):
    def get_adjacent_cells(cell):
        x, y = cell
        return ((x + i, y + j)
                for (i, j) in [(1, 0), (0, 1), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                if 0 <= x + i < dim
                if 0 <= y + j < dim), ((x + i, y + j)
                                       for (i, j) in [(1, 0), (0, 1), (0, -1)]
                                       if 0 <= x + i < dim
                                       if 0 <= y + j < dim)

    return get_adjacent_cells


def get_shortest_path(h_fun, grid):
    # Default start pos: (0,0)
    dim = len(grid[0])
    shortest_path = a_star_search((0, 0), get_neighbors(grid, dim), get_heuristic(h_fun, dim), grid)
    if shortest_path is None:
        return -1
    else:
        return shortest_path


if __name__ == '__main__':
    p = 0.1
    dim = 100
    grid = np.random.choice([0, 1], (dim * dim), p=[1 - p, p]).reshape(dim, dim)
    # dim = 4
    # grid = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    grid[0, 0] = 0
    n = {}
    c = {}
    b = {}
    e = {}
    h = {}
    blocked = {}
    grid[dim - 1, dim - 1] = 0
    print('Random Grid:')
    print(grid)
    start_manh = time.perf_counter()
    print('Manhattan Path:', get_shortest_path('MANHATTAN', grid))
    end_manh = time.perf_counter()
    print('Manhattan time:', round(end_manh - start_manh, 5))
    print('\n')

    print('* Note: -1 represents there is no path found from S to G')
