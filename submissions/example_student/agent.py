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
    Pacman (Seek Agent): Sử dụng thuật toán Minimax với cắt tỉa Alpha-Beta.
    Mục tiêu: Đánh giá trước các bước di chuyển, thu hẹp khoảng cách với Ghost 
    và dồn Ghost vào các vị trí ít đường lui (góc chết/ngõ cụt).
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "AlphaBeta_Seeker_Pacman"
        # Tốc độ tối đa của Pacman khi đi thẳng (Mặc định: 2)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.last_known_enemy_pos = None
        # Độ sâu của cây Minimax (Giới hạn ở mức 3 hoặc 4 để đảm bảo phản hồi dưới 1 giây)
        self.search_depth = 3
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        return 0 <= row < height and 0 <= col < width and map_state[row, col] == 0

    def _get_pacman_moves(self, pos: tuple, map_state: np.ndarray):
        """
        Lấy danh sách các nước đi hợp lệ của Pacman.
        Tận dụng lợi thế đi thẳng tối đa số ô cho phép (không rẽ chữ L).
        """
        moves = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = m.value
            for steps in range(1, self.pacman_speed + 1):
                # Đảm bảo toàn bộ các ô trên đường đi thẳng đều không có tường
                valid_straight_line = True
                for i in range(1, steps + 1):
                    inter_pos = (pos[0] + dr * i, pos[1] + dc * i)
                    if not self._is_valid_position(inter_pos, map_state):
                        valid_straight_line = False
                        break
                
                if valid_straight_line:
                    moves.append((m, steps))
                    
        if not moves:
            moves.append((Move.STAY, 1))
        return moves

    def _get_ghost_moves(self, pos: tuple, map_state: np.ndarray):
        """Lấy danh sách các nước đi hợp lệ của Ghost (1 bước)."""
        moves = [Move.STAY]
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = m.value
            nxt_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(nxt_pos, map_state):
                moves.append(m)
        return moves

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _evaluation_function(self, pacman_pos: tuple, ghost_pos: tuple, map_state: np.ndarray) -> float:
        """
        Hàm đánh giá điểm của một trạng thái.
        Điểm cao (Pacman ưu tiên) khi: Khoảng cách ngắn + Ghost bị dồn vào ngõ cụt.
        """
        dist = self._manhattan_distance(pacman_pos, ghost_pos)
        
        # Nếu khoảng cách < 2, Pacman đã bắt được Ghost (Điều kiện thắng) -> Điểm tuyệt đối
        if dist < 2:
            return 999999.0
            
        # Trừ điểm dựa trên khoảng cách (khoảng cách càng xa điểm càng âm)
        score = -dist * 10.0
        
        # Đếm số đường thoát của Ghost, số đường thoát càng nhiều thì trạng thái càng bất lợi cho Pacman
        ghost_escape_routes = self._get_ghost_moves(ghost_pos, map_state)
        score -= len(ghost_escape_routes) * 5.0
        
        return score

    def _alpha_beta(self, depth: int, agent_index: int, pacman_pos: tuple, ghost_pos: tuple, alpha: float, beta: float, map_state: np.ndarray):
        """
        Thuật toán Minimax kết hợp Alpha-Beta Pruning.
        agent_index = 0: Lượt của Pacman (Maximizer)
        agent_index = 1: Lượt của Ghost (Minimizer)
        """
        # Điều kiện dừng: Hết độ sâu hoặc đã chạm vào Ghost
        if depth == 0 or self._manhattan_distance(pacman_pos, ghost_pos) < 2:
            return self._evaluation_function(pacman_pos, ghost_pos, map_state), None

        if agent_index == 0:  # Lượt Pacman (Tối đa hóa điểm số)
            v = float('-inf')
            best_action = (Move.STAY, 1)
            
            for action in self._get_pacman_moves(pacman_pos, map_state):
                m, steps = action
                dr, dc = m.value
                new_pacman_pos = (pacman_pos[0] + dr * steps, pacman_pos[1] + dc * steps)
                
                score, _ = self._alpha_beta(depth, 1, new_pacman_pos, ghost_pos, alpha, beta, map_state)
                
                if score > v:
                    v = score
                    best_action = action
                    
                alpha = max(alpha, v)
                if alpha >= beta:
                    break # Cắt tỉa Beta
                    
            return v, best_action
            
        else:  # Lượt Ghost (Tối thiểu hóa điểm số của Pacman)
            v = float('inf')
            
            for action in self._get_ghost_moves(ghost_pos, map_state):
                if action == Move.STAY:
                    new_ghost_pos = ghost_pos
                else:
                    dr, dc = action.value
                    new_ghost_pos = (ghost_pos[0] + dr, ghost_pos[1] + dc)
                
                score, _ = self._alpha_beta(depth - 1, 0, pacman_pos, new_ghost_pos, alpha, beta, map_state)
                
                if score < v:
                    v = score
                    
                beta = min(beta, v)
                if alpha >= beta:
                    break # Cắt tỉa Alpha
                    
            return v, None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        target = enemy_position or self.last_known_enemy_pos
        
        # Nếu chưa có thông tin kẻ địch -> Đi dạo ngẫu nhiên (hoặc quét map)
        if target is None:
            moves = self._get_pacman_moves(my_position, map_state)
            if moves:
                # Tránh trường hợp chỉ có Move.STAY
                real_moves = [m for m in moves if m[0] != Move.STAY]
                if real_moves:
                    return random.choice(real_moves)
                return moves[0]
            return (Move.STAY, 1)

        # Chạy thuật toán Alpha-Beta để tìm nước đi tối ưu
        _, best_move = self._alpha_beta(
            depth=self.search_depth, 
            agent_index=0, 
            pacman_pos=my_position, 
            ghost_pos=target, 
            alpha=float('-inf'), 
            beta=float('inf'), 
            map_state=map_state
        )
        
        return best_move


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
                    return m
            return Move.STAY

        # 1. Chạy BFS từ vị trí của Pacman để lấy khoảng cách thực tế đến mọi ô trống
        pacman_distances = self._compute_bfs_distances(threat, map_state)
        
        best_move = Move.STAY
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
                    
        return best_move