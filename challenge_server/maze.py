"""Recursive-backtracker maze generator for challenge sessions."""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple


Cell = Tuple[int, int]


def generate_maze(cols: int, rows: int, rng: random.Random) -> dict:
    cols = max(5, cols)
    rows = max(5, rows)

    h_walls = [[True] * cols for _ in range(rows + 1)]
    v_walls = [[True] * (cols + 1) for _ in range(rows)]

    visited = [[False] * cols for _ in range(rows)]
    stack: List[Cell] = [(0, 0)]
    visited[0][0] = True

    while stack:
        cx, cy = stack[-1]
        options: List[Tuple[int, int, str]] = []
        if cx > 0 and not visited[cy][cx - 1]:
            options.append((cx - 1, cy, "w"))
        if cx < cols - 1 and not visited[cy][cx + 1]:
            options.append((cx + 1, cy, "e"))
        if cy > 0 and not visited[cy - 1][cx]:
            options.append((cx, cy - 1, "n"))
        if cy < rows - 1 and not visited[cy + 1][cx]:
            options.append((cx, cy + 1, "s"))

        if not options:
            stack.pop()
            continue

        nx, ny, direction = rng.choice(options)
        if direction == "w":
            v_walls[cy][cx] = False
        elif direction == "e":
            v_walls[ny][cx + 1] = False
        elif direction == "n":
            h_walls[cy][cx] = False
        else:
            h_walls[ny + 1][cx] = False

        visited[ny][nx] = True
        stack.append((nx, ny))

    h_walls[0][0] = False
    h_walls[rows][cols - 1] = False
    v_walls[0][0] = False
    v_walls[rows - 1][cols] = False

    return {
        "cols": cols,
        "rows": rows,
        "h_walls": h_walls,
        "v_walls": v_walls,
        "start_cell": [0, 0],
        "end_cell": [cols - 1, rows - 1],
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

    return [start, end]
