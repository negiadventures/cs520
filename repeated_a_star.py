import math
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
        if h_fun == 'MANHATTAN':
            return (abs(dim - i) + abs(dim - j))*2
        elif h_fun == 'MANHATTAN_3':
            return (abs(dim - i) + abs(dim - j))*3
        elif h_fun == 'EUCLIDEAN':
            return math.sqrt(abs(dim - i) ** 2 + abs(dim - j) ** 2)
        elif h_fun == 'CHEBYSHEV':
            return max(abs(dim - i), abs(dim - j))
        else:
            # DEFAULT: MANHATTAN
            return abs(dim - i) + abs(dim - j)

    return calc_h


def repeated_a_star_search_q6(start, neighbors, heuristic, grid, blocked, parent):
    global cells_processed
    visited = set()
    distance = {start: 0}
    fringe = PriorityQueue()
    fringe.add(start)
    # print(start)
    while fringe:
        cell = fringe.pop()
        cells_processed += 1
        x, y = cell
        if cell in visited:
            continue
        if cell == (dim - 1, dim - 1):
            return repeated_a_star_reconstruct_path(parent, start, cell)
        if grid[x][y] == 1:
            print('Reconstruct')
            return repeated_a_star_reconstruct_path(parent, start, cell)
        visited.add(cell)
        for child in neighbors(cell, blocked):
            fringe.add(child, priority=distance[cell] + 1 + heuristic(child))
            if child not in distance or distance[cell] + 1 < distance[child]:
                distance[child] = distance[cell] + 1
                parent[child] = cell
    return []


def repeated_a_star_reconstruct_path(parent, start, end):
    if grid[end[0]][end[1]] != 1:
        path = [end]
    else:
        path = []
    while end != start:
        end = parent[end]
        path.append(end)
    return list(reversed(path))


def repeated_a_star_get_neighbors_q6(grid, dim):
    def get_adjacent_cells(cell, blocked):
        x, y = cell
        for (i, j) in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
            if 0 <= x + i < dim:
                if 0 <= y + j < dim:
                    if (x + i, y + j) not in blocked:
                        if grid[x + i][y + j] == 1:
                            blocked.append((x + i, y + j))
        return ((x + i, y + j)
                for (i, j) in [(1, 0), (0, 1), (0, -1), (-1, 0)]
                # (i, j) Represents movement from current cell - N,W,S,E direction eg: (1,0) means -> (x+1, y)
                # neighbor should be within grid boundary
                # neighbor should be an unblocked cell
                if 0 <= x + i < dim
                if 0 <= y + j < dim
                if (grid[x + i][y + j] == 0 and grid[x][y] != 1) or (grid[x + i][y + j] == 1 and grid[x][y] != 1)
                if (x + i, y + j) not in blocked
                )

    return get_adjacent_cells


def repeated_a_star_get_shortest_path_q6(h_fun, grid):
    global trajectory_length
    dim = len(grid[0])
    path = []
    start = (0, 0)
    shortest_path = []
    blocked = []
    parent = dict()
    while (dim - 1, dim - 1) not in shortest_path:
        print('repeat')
        shortest_path = repeated_a_star_search_q6(start, repeated_a_star_get_neighbors_q6(grid, dim),
                                                  get_heuristic(h_fun, dim), grid, blocked, parent)
        if len(shortest_path) == 0:
            return -1
        path.extend(shortest_path)
        try:
            start = shortest_path[len(shortest_path) - 1]
        except:
            pass
    if (dim - 1, dim - 1) not in path:
        return -1
    else:
        trajectory_length = len(path)
        print('length:', len(path))
        return path


if __name__ == '__main__':
    p = 0.25
    dim = 101
    grid = np.random.choice([0, 1], (dim * dim), p=[1 - p, p]).reshape(dim, dim)
    grid[0, 0] = 0
    grid[dim - 1, dim - 1] = 0
    cells_processed = 0
    trajectory_length = 0
    print('Random Grid:')
    print(grid)

    # start_manh = time.perf_counter()
    # print('Manhattan Path:', get_shortest_path('MANHATTAN', grid))
    # end_manh = time.perf_counter()
    # print('Manhattan time:', round(end_manh - start_manh, 5))

    start_manh = time.perf_counter()
    print('Manhattan Path:', repeated_a_star_get_shortest_path_q6('m', grid))
    end_manh = time.perf_counter()
    print('Manhattan time:', round(end_manh - start_manh, 5))
    print('\n')

    start_manh = time.perf_counter()
    print('Inadmissible Manhattan(x2) Path:', repeated_a_star_get_shortest_path_q6('MANHATTAN', grid))
    end_manh = time.perf_counter()
    print('Inadmissible Manhattan(x2) time:', round(end_manh - start_manh, 5))
    print('\n')

    start_manh = time.perf_counter()
    print('Inadmissible Manhattan(x3) Path:', repeated_a_star_get_shortest_path_q6('MANHATTAN_3', grid))
    end_manh = time.perf_counter()
    print('Inadmissible Manhattan(x3) time:', round(end_manh - start_manh, 5))
    print('\n')

    print('cells processed:', cells_processed)
    print('trajectory length:', trajectory_length)


    print('* Note: -1 represents there is no path found from S to G')
