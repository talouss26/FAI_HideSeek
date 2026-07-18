# src/tournament_runner.py
import subprocess
import re
import os
from pathlib import Path

def verify_agent_paths(agent_ids):
    """Kiểm tra sự tồn tại của các tệp agent trước khi chạy giải đấu."""
    root_dir = Path(__file__).parent.parent.absolute()
    submissions_dir = root_dir / "submissions"
    
    all_valid = True
    print("\n🔍 [KIỂM TRA CẤU TRÚC PHÂN HỆ AGENT]")
    for agent_id in agent_ids:
        agent_folder = submissions_dir / agent_id
        agent_file = agent_folder / "agent.py"
        
        if not agent_folder.exists():
            print(f"❌ Không tìm thấy THƯ MỤC: {agent_folder}")
            all_valid = False
        elif not agent_file.exists():
            print(f"❌ Thư mục tồn tại nhưng SAI TÊN FILE (Phải là agent.py): {agent_file}")
            # Quét tìm xem có file đặt sai tên dạng 2agent.py, 4agent.py không
            existing_files = list(agent_folder.glob("*.py"))
            if existing_files:
                print(f"   👉 Phát hiện các file đang có trong thư mục: {[f.name for f in existing_files]}")
            all_valid = False
        else:
            print(f"✅ Hợp lệ: submissions/{agent_id}/agent.py")
            
    return all_valid

def run_single_match(seek_id, hide_id, debug=False):
    src_dir = Path(__file__).parent.absolute()
    
    cmd = [
        "python", "arena.py",
        "--seek", seek_id,
        "--hide", hide_id,
        "--no-viz",
        "--start-mode", "stochastic",
        "--pacman-speed", "2"
    ]
    
    env_config = os.environ.copy()
    env_config["PYTHONIOENCODING"] = "utf-8"
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        encoding='utf-8', 
        cwd=str(src_dir), 
        env=env_config
    )
    output = result.stdout + "\n" + result.stderr
    
    if debug and result.returncode != 0:
        print(f"\n[DEBUG LOG TRẬN ĐẤU ĐÃ BỊ LỖI ({seek_id} VS {hide_id})]:")
        print(output)
        print("-" * 60)
    
    winner = "draw"
    if "🏆 WINNER:" in output:
        if "(Pacman)" in output:
            winner = "pacman"
        elif "(Ghost)" in output:
            winner = "ghost"
            
    steps = 200
    steps_match = re.search(r"Total Steps:\s*(\d+)", output)
    if steps_match:
        steps = int(steps_match.group(1))
        
    return winner, steps

def evaluate_team_performance(my_team_id, opponent_pool, num_matches=5):
    # Xác thực cấu trúc tệp trước khi cấp quyền chạy tiến trình con
    all_agents = [my_team_id] + opponent_pool
    if not verify_agent_paths(all_agents):
        print("\n🛑 Quy trình dừng lại: Vui lòng sửa cấu trúc hoặc tên tệp agent theo cảnh báo phía trên.")
        return

    print(f"\n============================================================")
    print(f"🚀 BẮT ĐẦU QUY TRÌNH KIỂM THỬ VÀ ĐÁNH GIÁ CHỈ SỐ PHỤ")
    print(f"👥 Đội nhà: {my_team_id} | 🤖 Quân xanh thử nghiệm: {opponent_pool}")
    print(f"============================================================")
    
    pacman_wins, pacman_total_steps, pacman_match_count = 0, 0, 0
    ghost_wins, ghost_total_steps, ghost_match_count = 0, 0, 0
    
    print("\n[🛡️ LƯỢT 1] Đội nhà làm PACMAN đi săn Quân xanh...")
    for opponent in opponent_pool:
        for match in range(num_matches):
            winner, steps = run_single_match(my_team_id, opponent)
            if winner == "pacman":
                pacman_wins += 1
            pacman_total_steps += steps
            pacman_match_count += 1
            
    avg_steps_as_pacman = pacman_total_steps / pacman_match_count
    pacman_win_rate = (pacman_wins / pacman_match_count) * 100
    
    print("[🥷 LƯỢT 2] Đội nhà làm GHOST trốn Quân xanh...")
    for opponent in opponent_pool:
        for match in range(num_matches):
            winner, steps = run_single_match(opponent, my_team_id)
            if winner == "ghost":
                ghost_wins += 1
            ghost_total_steps += steps
            ghost_match_count += 1
            
    avg_steps_as_ghost = ghost_total_steps / ghost_match_count
    ghost_win_rate = (ghost_wins / ghost_match_count) * 100
    
    tie_breaking_score = avg_steps_as_ghost - avg_steps_as_pacman
    
    print(f"\n============================================================")
    print(f"📊 BÁO CÁO HIỆU NĂNG TỔNG HỢP CỦA THÀNH VIÊN 3")
    print(f"============================================================")
    print(f"🥇 Tỷ lệ thắng khi làm Seeker (Pacman): {pacman_win_rate:.2f}%")
    print(f"⏱️ Bước đi trung bình khi làm Pacman   : {avg_steps_as_pacman:.2f} bước")
    print(f"------------------------------------------------------------")
    print(f"🥈 Tỷ lệ thắng khi làm Hider (Ghost)   : {ghost_win_rate:.2f}%")
    print(f"⏱️ Bước đi trung bình khi làm Ghost    : {avg_steps_as_ghost:.2f} bước")
    print(f"------------------------------------------------------------")
    print(f"🏆 ĐỘ CHÊNH LỆCH BƯỚC ĐI (TIE-BREAKING): {tie_breaking_score:+.2f} bước")
    print(f"============================================================\n")

if __name__ == "__main__":
    MY_TEAM = "example_student" 
    QUAN_XANH_POOL = ["A", "B", "C", "simple_agent"]
    RUN_MATCHES = 5 
    
    evaluate_team_performance(my_team_id=MY_TEAM, opponent_pool=QUAN_XANH_POOL, num_matches=RUN_MATCHES)