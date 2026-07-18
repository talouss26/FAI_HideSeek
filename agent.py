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
    Ghost (Hide Agent): Sử dụng thuật toán Minimax kết hợp Alpha-Beta Pruning.
    Mục tiêu: Đóng vai trò Maximizer để tối đa hóa khoảng cách thực tế (BFS) với Pacman
    và chủ động tránh né các góc chết, ngõ cụt trên bản đồ.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "AlphaBeta_Evasive_Ghost"
        self.last_known_enemy_pos = None
        # Độ sâu cây tìm kiếm (Ghost chỉ đi 1 ô nên depth=4 hoặc 5 vẫn chạy cực mượt dưới 1s)
        self.search_depth = 4 

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        return 0 <= row < height and 0 <= col < width and map_state[row, col] == 0

    def _compute_bfs_distances(self, start: tuple, map_state: np.ndarray) -> dict:
        """Tính khoảng cách bước đi thực tế ngắn nhất từ một vị trí đến mọi ô trên bản đồ."""
        distances = {start: 0}
        queue = deque([start])
        
        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, col_diff = move.value
                nxt = (curr[0] + dr, curr[1] + col_diff)
                if self._is_valid_position(nxt, map_state) and nxt not in distances:
                    distances[nxt] = curr_dist + 1
                    queue.append(nxt)
        return distances

    def _get_ghost_moves(self, pos: tuple, map_state: np.ndarray):
        """Ghost đi tối đa 1 ô hoặc đứng yên."""
        moves = [Move.STAY]
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = m.value
            nxt_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(nxt_pos, map_state):
                moves.append(m)
        return moves

    def _get_pacman_moves(self, pos: tuple, map_state: np.ndarray) -> list:
        """Giả định luật di chuyển đi thẳng (tối đa 2 ô) của Pacman để tính toán."""
        moves = []
        # Giả định tốc độ thông thường trong Arena là 2
        pacman_speed = 2 
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = m.value
            for steps in range(1, pacman_speed + 1):
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

    def _evaluation_function(self, ghost_pos: tuple, pacman_pos: tuple, pacman_distances: dict, map_state: np.ndarray) -> float:
        """
        Hàm đánh giá trạng thái dưới góc nhìn của Ghost (Muốn điểm CÀNG CAO CÀNG TỐT).
        """
        # Khoảng cách Manhattan
        manhattan_dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        
        # Nếu Pacman bắt được Ghost (Khoảng cách Manhattan < 2) -> Điểm phạt cực nặng
        if manhattan_dist < 2:
            return -999999.0

        # Lấy khoảng cách đường đi thực tế (BFS) từ Pacman tới Ghost
        bfs_dist = pacman_distances.get(ghost_pos, manhattan_dist)
        
        # Điểm cơ bản dựa trên khoảng cách thực tế (Càng xa càng tốt)
        score = bfs_dist * 50.0
        
        # Phạt nặng nếu Ghost tự chui vào các ngõ cụt (ít đường thoát)
        ghost_escape_routes = self._get_ghost_moves(ghost_pos, map_state)
        # Số lượng đường đi càng ít thì Ghost càng dễ bị dồn góc
        if len(ghost_escape_routes) <= 2:  # Chỉ có 1 đường đi tiếp + 1 đứng yên
            score -= 150.0
            
        return score

    def _alpha_beta(self, depth: int, agent_index: int, ghost_pos: tuple, pacman_pos: tuple, 
                    alpha: float, beta: float, pacman_distances: dict, map_state: np.ndarray):
        """
        Minimax Alpha-Beta:
        agent_index = 0: Ghost (Maximizer - Trốn xa nhất)
        agent_index = 1: Pacman (Minimizer - Thu hẹp khoảng cách)
        """
        # Điều kiện dừng
        manhattan_dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        if depth == 0 or manhattan_dist < 2:
            return self._evaluation_function(ghost_pos, pacman_pos, pacman_distances, map_state), None

        if agent_index == 0:  # Lượt của Ghost (Maximizer)
            v = float('-inf')
            best_action = Move.STAY
            
            # Xáo trộn nước đi tránh bị lặp vòng vô tận vô nghĩa
            moves = self._get_ghost_moves(ghost_pos, map_state)
            random.shuffle(moves)
            
            for action in moves:
                if action == Move.STAY:
                    nxt_ghost_pos = ghost_pos
                else:
                    dr, dc = action.value
                    nxt_ghost_pos = (ghost_pos[0] + dr, ghost_pos[1] + dc)
                    
                score, _ = self._alpha_beta(depth - 1, 1, nxt_ghost_pos, pacman_pos, alpha, beta, pacman_distances, map_state)
                
                if score > v:
                    v = score
                    best_action = action
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v, best_action

        else:  # Lượt của Pacman (Minimizer)
            v = float('inf')
            
            for action in self._get_pacman_moves(pacman_pos, map_state):
                m, steps = action
                dr, dc = m.value
                nxt_pacman_pos = (pacman_pos[0] + dr * steps, pacman_pos[1] + dc * steps)
                
                score, _ = self._alpha_beta(depth - 1, 0, ghost_pos, nxt_pacman_pos, alpha, beta, pacman_distances, map_state)
                
                if score < v:
                    v = score
                beta = min(beta, v)
                if alpha >= beta:
                    break
            return v, None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        threat = enemy_position or self.last_known_enemy_pos
        
        # Nếu hoàn toàn mất dấu Pacman -> Di chuyển ngẫu nhiên tìm đường thông thoáng
        if threat is None:
            moves = [m for m in self._get_ghost_moves(my_position, map_state) if m != Move.STAY]
            return random.choice(moves) if moves else Move.STAY

        # Tính trước ma trận khoảng cách BFS từ vị trí hiện tại của Pacman làm tài nguyên đánh giá
        pacman_distances = self._compute_bfs_distances(threat, map_state)

        # Chạy Minimax Alpha-Beta để tìm nước đi né tránh tối ưu nhìn xa trông rộng
        _, best_move = self._alpha_beta(
            depth=self.search_depth,
            agent_index=0,
            ghost_pos=my_position,
            pacman_pos=threat,
            alpha=float('-inf'),
            beta=float('inf'),
            pacman_distances=pacman_distances,
            map_state=map_state
        )
        
        return best_move