"""Recursive-backtracker maze generator — guaranteed solvable from start to end."""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple


Cell = Tuple[int, int]


def _build_wall_grid(cols: int, rows: int, rng: random.Random) -> tuple:
    """Each cell has north/east/south/west walls (True = wall present)."""
    north = [[True] * cols for _ in range(rows)]
    east = [[True] * cols for _ in range(rows)]
    south = [[True] * cols for _ in range(rows)]
    west = [[True] * cols for _ in range(rows)]

    visited = [[False] * cols for _ in range(rows)]
    stack: List[Cell] = [(0, 0)]
    visited[0][0] = True

    while stack:
        cx, cy = stack[-1]
        neighbors: List[Tuple[int, int, str]] = []
        if cx > 0 and not visited[cy][cx - 1]:
            neighbors.append((cx - 1, cy, "w"))
        if cx < cols - 1 and not visited[cy][cx + 1]:
            neighbors.append((cx + 1, cy, "e"))
        if cy > 0 and not visited[cy - 1][cx]:
            neighbors.append((cx, cy - 1, "n"))
        if cy < rows - 1 and not visited[cy + 1][cx]:
            neighbors.append((cx, cy + 1, "s"))

        if not neighbors:
            stack.pop()
            continue

        nx, ny, direction = rng.choice(neighbors)
        if direction == "w":
            west[cy][cx] = False
            east[cy][nx] = False
        elif direction == "e":
            east[cy][cx] = False
            west[ny][nx] = False
        elif direction == "n":
            north[cy][cx] = False
            south[ny][nx] = False
        else:
            south[cy][cx] = False
            north[ny][nx] = False

        visited[ny][nx] = True
        stack.append((nx, ny))

    # Convert to legacy h_walls / v_walls layout used by the canvas renderer.
    h_walls = [[True] * cols for _ in range(rows + 1)]
    v_walls = [[True] * (cols + 1) for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if not north[r][c]:
                h_walls[r][c] = False
            if not south[r][c]:
                h_walls[r + 1][c] = False
            if not west[r][c]:
                v_walls[r][c] = False
            if not east[r][c]:
                v_walls[r][c + 1] = False

    return h_walls, v_walls


def generate_maze(cols: int, rows: int, rng: random.Random) -> dict:
    cols = max(5, cols)
    rows = max(5, rows)
    start_cell = [0, 0]
    end_cell = [cols - 1, rows - 1]

    for _ in range(40):
        h_walls, v_walls = _build_wall_grid(cols, rows, rng)
        maze = {
            "cols": cols,
            "rows": rows,
            "h_walls": h_walls,
            "v_walls": v_walls,
            "start_cell": start_cell,
            "end_cell": end_cell,
        }
        path = solve_maze_path(maze)
        if len(path) >= 2 and path[-1] == tuple(end_cell):
            return maze

    # Fallback: open a simple corridor along the bottom and right edges.
    h_walls, v_walls = _build_wall_grid(cols, rows, rng)
    for c in range(cols):
        h_walls[rows][c] = False
    for r in range(rows):
        v_walls[r][cols] = False
    return {
        "cols": cols,
        "rows": rows,
        "h_walls": h_walls,
        "v_walls": v_walls,
        "start_cell": start_cell,
        "end_cell": end_cell,
    }


def maze_cell_centers(
    maze: dict,
    origin_x: float,
    origin_y: float,
    cell_w: float,
    cell_h: float,
) -> Tuple[List[float], List[float]]:
    sc = maze["start_cell"]
    ec = maze["end_cell"]
    start = [origin_x + (sc[0] + 0.5) * cell_w, origin_y + (sc[1] + 0.5) * cell_h]
    end = [origin_x + (ec[0] + 0.5) * cell_w, origin_y + (ec[1] + 0.5) * cell_h]
    return start, end


def maze_path_pixels(maze: dict, origin_x: float, origin_y: float, cell_w: float, cell_h: float) -> List[List[float]]:
    path = solve_maze_path(maze)
    return [
        [origin_x + (cx + 0.5) * cell_w, origin_y + (cy + 0.5) * cell_h]
        for cx, cy in path
    ]


def solve_maze_path(maze: dict) -> List[Cell]:
    cols, rows = maze["cols"], maze["rows"]
    h_walls, v_walls = maze["h_walls"], maze["v_walls"]
    start = tuple(maze["start_cell"])
    end = tuple(maze["end_cell"])

    queue: deque = deque([(start, [start])])
    seen = {start}

    while queue:
        (cx, cy), path = queue.popleft()
        if (cx, cy) == end:
            return path

        for nx, ny, can_move in (
            (cx - 1, cy, not v_walls[cy][cx]),
            (cx + 1, cy, not v_walls[cy][cx + 1]),
            (cx, cy - 1, not h_walls[cy][cx]),
            (cx, cy + 1, not h_walls[cy + 1][cx]),
        ):
            if 0 <= nx < cols and 0 <= ny < rows and can_move and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

    return [start]
