import sys
from pathlib import Path
from collections import deque
import random
import numpy as np

# Thêm src vào hệ thống để import interface từ framework
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seek Agent): Chỉ sử dụng thuật toán BFS để tìm đường đi ngắn nhất 
    đến vị trí của Ghost nhằm bắt mục tiêu nhanh nhất.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS_Seeker_Pacman"
        # Tốc độ tối đa của Pacman khi đi thẳng [cite: 73, 82]
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.last_known_enemy_pos = None
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        return 0 <= row < height and 0 <= col < width and map_state[row, col] == 0

    def _bfs_shortest_path(self, start: tuple, goal: tuple, map_state: np.ndarray):
        """Tìm hướng đi đầu tiên (Move) trên đường đi ngắn nhất từ start đến goal bằng BFS."""
        if start == goal:
            return None
            
        queue = deque([(start, None)])  # (vị trí hiện tại, hướng đi đầu tiên)
        visited = {start}
        
        while queue:
            curr_pos, first_move = queue.popleft()
            
            if curr_pos == goal:
                return first_move
                
            # Duyệt các hướng di chuyển có thể
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                
                if self._is_valid_position(nxt_pos, map_state) and nxt_pos not in visited:
                    visited.add(nxt_pos)
                    # Nếu chưa có hướng đi đầu tiên (nút kề gốc), gán bằng move hiện tại
                    nxt_first_move = first_move if first_move is not None else move
                    queue.append((nxt_pos, nxt_first_move))
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        """Tính số bước đi thẳng tối đa mà không đâm vào tường[cite: 82]."""
        steps = 0
        current = pos
        for _ in range(max_steps):
            dr, dc = move.value
            next_pos = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        target = enemy_position or self.last_known_enemy_pos
        
        # Nếu không có thông tin kẻ địch -> Đi ngẫu nhiên để dò tìm [cite: 57]
        if target is None:
            all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(all_moves)
            for m in all_moves:
                steps = self._max_valid_steps(my_position, m, map_state, self.pacman_speed)
                if steps > 0:
                    return (m, steps)
            return (Move.STAY, 1)

        # Sử dụng BFS để tìm hướng đi ngắn nhất đến vị trí Ghost
        best_move = self._bfs_shortest_path(my_position, target, map_state)
        
        if best_move is not None:
            # Di chuyển thẳng tối đa có thể theo hướng đó (lên tới giới hạn pacman_speed) [cite: 73, 82]
            steps = self._max_valid_steps(my_position, best_move, map_state, self.pacman_speed)
            if steps > 0:
                return (best_move, steps) [cite: 80]
                
        # Phương án dự phòng nếu bị kẹt hoặc lỗi đồ thị
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for m in all_moves:
            steps = self._max_valid_steps(my_position, m, map_state, 1)
            if steps > 0:
                return (m, steps) [cite: 80]
                
        return (Move.STAY, 1) [cite: 80]


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hide Agent): Sử dụng thuật toán BFS để quét toàn bộ bản đồ từ vị trí Pacman, 
    sau đó chọn bước đi giúp tối đa hóa khoảng cách thực tế nhằm trốn lâu nhất.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS_Evasive_Ghost"
        self.last_known_enemy_pos = None

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        return 0 <= row < height and 0 <= col < width and map_state[row, col] == 0

    def _compute_bfs_distances(self, start: tuple, map_state: np.ndarray) -> dict:
        """Dùng BFS loang toàn bộ bản đồ để tính khoảng cách ngắn nhất từ 'start' đến mọi ô."""
        distances = {start: 0}
        queue = deque([start])
        
        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt = (curr[0] + dr, curr[1] + dc)
                if self._is_valid_position(nxt, map_state) and nxt not in distances:
                    distances[nxt] = curr_dist + 1
                    queue.append(nxt)
        return distances

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        threat = enemy_position or self.last_known_enemy_pos
        
        # Nếu không biết vị trí Pacman -> Di chuyển ngẫu nhiên
        if threat is None:
            all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(all_moves)
            for m in all_moves:
                dr, dc = m.value
                if self._is_valid_position((my_position[0] + dr, my_position[1] + dc), map_state):
                    return m [cite: 83]
            return Move.STAY [cite: 83]

        # 1. Chạy BFS từ vị trí của Pacman để lấy khoảng cách thực tế đến mọi ô trống
        pacman_distances = self._compute_bfs_distances(threat, map_state)
        
        best_move = Move.STAY [cite: 83]
        # Khoảng cách hiện tại từ Pacman tới vị trí đứng của Ghost
        max_distance_from_enemy = pacman_distances.get(my_position, 0)
        
        # 2. Thử mọi hướng đi xung quanh (bao gồm đứng yên) để xem hướng nào giữ khoảng cách xa nhất
        possible_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        random.shuffle(possible_moves)  # Xáo trộn tránh trường hợp bị kẹt đi qua đi lại một chỗ
        
        for move in possible_moves:
            if move == Move.STAY:
                continue
            
            dr, dc = move.value
            next_pos = (my_position[0] + dr, my_position[1] + dc)
            
            if self._is_valid_position(next_pos, map_state):
                dist_from_enemy = pacman_distances.get(next_pos, 0)
                
                # Ưu tiên bước đi tối đa hóa khoảng cách thực tế với Pacman
                if dist_from_enemy > max_distance_from_enemy:
                    max_distance_from_enemy = dist_from_enemy
                    best_move = move
                    
        return best_move [cite: 83]