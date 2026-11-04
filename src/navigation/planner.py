"""仅从 LiDAR 与估计位姿建立占据图；不读取障碍物真实位置。"""

import heapq
import math

import numpy as np


class OccupancyPlanner:
    def __init__(self, resolution=0.07, inflation=0.24):
        self.resolution = resolution
        self.origin = np.array([-1.5, -1.5])
        self.shape = (int(4 / resolution) + 1, int(3 / resolution) + 1)
        self.logodds = np.zeros(self.shape)
        self.inflation = inflation

    def cell(self, p):
        return tuple(
            np.floor((np.asarray(p) - self.origin) / self.resolution).astype(int)
        )

    def point(self, c):
        return self.origin + (np.asarray(c) + 0.5) * self.resolution

    def inside(self, c):
        return 0 <= c[0] < self.shape[0] and 0 <= c[1] < self.shape[1]

    @staticmethod
    def line(a, b):
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1])) + 1
        return list(
            dict.fromkeys(tuple(x) for x in np.rint(np.linspace(a, b, n)).astype(int))
        )

    def update(self, scan, pose):
        theta = pose[2]
        sensor = pose[:2] + np.array([-0.049 * np.cos(theta), -0.049 * np.sin(theta)])
        start = self.cell(sensor)
        for angle, distance in zip(scan.angles, scan.ranges):
            if (
                not np.isfinite(distance)
                or not scan.range_min <= distance <= scan.range_max
            ):
                continue
            end = sensor + distance * np.array(
                [np.cos(theta + angle), np.sin(theta + angle)]
            )
            cells = self.line(start, self.cell(end))
            for cell in cells[:-1]:
                if self.inside(cell):
                    self.logodds[cell] = max(-5, self.logodds[cell] - 0.35)
            if self.inside(cells[-1]):
                self.logodds[cells[-1]] = min(7, self.logodds[cells[-1]] + 1.4)

    def blocked(self):
        occupied = self.logodds > 0.7
        inflated = occupied.copy()
        radius = math.ceil(self.inflation / self.resolution)
        for x, y in np.argwhere(occupied):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if math.hypot(dx, dy) * self.resolution <= self.inflation + 0.04:
                        c = (x + dx, y + dy)
                        if self.inside(c):
                            inflated[c] = True
        return inflated

    def plan(self, start, goal):
        blocked = self.blocked()
        s = self.cell(start)
        g = self.cell(goal)
        if not self.inside(s) or not self.inside(g) or blocked[g]:
            return None
        # 自身所在格可能受小量定位误差影响，允许从当前格开始但不清除周围障碍。
        blocked[s] = False
        queue = [(0.0, s)]
        cost = {s: 0.0}
        parent = {}
        while queue:
            _, current = heapq.heappop(queue)
            if current == g:
                break
            for dx, dy in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                nxt = (current[0] + dx, current[1] + dy)
                if not self.inside(nxt) or blocked[nxt]:
                    continue
                if (
                    dx
                    and dy
                    and (
                        blocked[current[0] + dx, current[1]]
                        or blocked[current[0], current[1] + dy]
                    )
                ):
                    continue
                unknown = abs(self.logodds[nxt]) < 0.1
                candidate = cost[current] + math.hypot(dx, dy) * (1.4 if unknown else 1)
                if candidate < cost.get(nxt, float("inf")):
                    cost[nxt] = candidate
                    parent[nxt] = current
                    heapq.heappush(queue, (candidate + math.dist(nxt, g), nxt))
        if g not in cost:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        cells.reverse()
        # 安全直线捷径减少网格折线，但每条捷径仍经过膨胀地图检查。
        smooth = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and any(blocked[c] for c in self.line(cells[i], cells[j])):
                j -= 1
            smooth.append(cells[j])
            i = j
        points = np.array([self.point(c) for c in smooth])
        points[0] = start
        points[-1] = goal
        return points
