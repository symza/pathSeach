import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import heapq
import random
import os
import sys

# 获取脚本所在目录和dijkstra文件夹路径
# 处理 PyInstaller 打包后的情况
def get_base_path():
    """获取基础路径，兼容打包后的exe和普通脚本"""
    if getattr(sys, 'frozen', False):
        # 打包后的exe文件
        base_path = os.path.dirname(sys.executable)
    else:
        # 普通Python脚本
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return base_path

DIJKSTRA_DIR = get_base_path()  # dijkstra文件夹（exe所在目录或脚本的上级目录）
IMAGES_DIR = os.path.join(DIJKSTRA_DIR, 'images')  # images文件夹路径

# 确保images文件夹存在
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MazeDijkstra:
    def __init__(self, width=30, height=30, complexity=0.75, density=0.75):
        """
        创建迷宫并初始化Dijkstra算法
        
        参数:
            width: 迷宫宽度
            height: 迷宫高度
            complexity: 复杂度（0-1），值越大迷宫越复杂
            density: 密度（0-1），值越大墙壁越多
        """
        self.width = width
        self.height = height
        
        # 先设置起点和终点
        self.start = (1, 1)
        self.end = (height - 2, width - 2)
        
        # 生成迷宫
        self.maze = self.generate_maze(width, height, complexity, density)
        
        # 确保起点和终点是通路
        self.maze[self.start[0], self.start[1]] = 0
        self.maze[self.end[0], self.end[1]] = 0
        
        # 初始化距离字典
        self.distances = {}
        self.previous = {}
        self.visited = set()
        self.steps = []
        
        # 执行Dijkstra算法并记录每一步
        self.run_dijkstra()
        
        # 扩展步骤以增加帧数
        self.expand_steps()
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(14, 14), dpi=150)
        
    def generate_maze(self, width, height, complexity, density):
        """生成复杂迷宫"""
        # 初始化：全部是墙壁
        maze = np.ones((height, width), dtype=int)
        
        # 创建基础路径网格
        for i in range(1, height - 1, 2):
            for j in range(1, width - 1, 2):
                maze[i, j] = 0  # 创建基础通路
        
        # 添加随机连接
        for i in range(1, height - 1, 2):
            for j in range(1, width - 1, 2):
                if random.random() < complexity:
                    # 随机选择方向
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    random.shuffle(directions)
                    for di, dj in directions[:2]:  # 尝试两个方向
                        ni, nj = i + di, j + dj
                        if 0 < ni < height - 1 and 0 < nj < width - 1:
                            maze[ni, nj] = 0
        
        # 添加额外的随机墙壁（增加复杂度）
        num_walls = int(density * (width * height) * 0.3)
        for _ in range(num_walls):
            i, j = random.randint(1, height - 2), random.randint(1, width - 2)
            if (i, j) != self.start and (i, j) != self.end:
                maze[i, j] = 1
        
        # 确保有路径连接起点和终点（使用BFS验证）
        if not self.has_path(maze, self.start, self.end):
            # 如果没有路径，创建一条
            self.create_path(maze, self.start, self.end)
        
        return maze
    
    def has_path(self, maze, start, end):
        """使用BFS检查是否存在路径"""
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            if current == end:
                return True
            
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = current[0] + di, current[1] + dj
                if (0 <= ni < self.height and 0 <= nj < self.width and 
                    (ni, nj) not in visited and maze[ni, nj] == 0):
                    visited.add((ni, nj))
                    queue.append((ni, nj))
        
        return False
    
    def create_path(self, maze, start, end):
        """创建从起点到终点的路径"""
        current = start
        while current != end:
            di = 1 if end[0] > current[0] else (-1 if end[0] < current[0] else 0)
            dj = 1 if end[1] > current[1] else (-1 if end[1] < current[1] else 0)
            
            if di != 0:
                current = (current[0] + di, current[1])
            elif dj != 0:
                current = (current[0], current[1] + dj)
            else:
                break
            
            if 0 < current[0] < self.height - 1 and 0 < current[1] < self.width - 1:
                maze[current[0], current[1]] = 0
    
    def run_dijkstra(self):
        """执行Dijkstra算法并记录每一步"""
        # 初始化
        pq = [(0, self.start)]  # (distance, node)
        self.distances = {self.start: 0}
        self.previous = {}
        self.visited = set()
        processed_nodes = set()
        
        # 记录初始状态
        self.steps.append({
            'distances': self.distances.copy(),
            'current_node': None,
            'visited': self.visited.copy(),
            'processed': processed_nodes.copy(),
            'previous': self.previous.copy(),
            'message': f'初始化：起点 {self.start} 距离为0'
        })
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in self.visited:
                continue
            
            self.visited.add(current_node)
            processed_nodes.add(current_node)
            
            # 记录选择当前节点
            self.steps.append({
                'distances': self.distances.copy(),
                'current_node': current_node,
                'visited': self.visited.copy(),
                'processed': processed_nodes.copy(),
                'previous': self.previous.copy(),
                'message': f'选择节点 {current_node}（距离：{current_dist}）'
            })
            
            # 如果到达终点，提前结束
            if current_node == self.end:
                break
            
            # 检查邻居节点（上下左右）
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current_node[0] + di, current_node[1] + dj)
                ni, nj = neighbor
                
                # 检查边界和墙壁
                if (ni < 0 or ni >= self.height or nj < 0 or nj >= self.width or
                    self.maze[ni, nj] == 1 or neighbor in self.visited):
                    continue
                
                # 记录检查邻居的过程
                self.steps.append({
                    'distances': self.distances.copy(),
                    'current_node': current_node,
                    'visited': self.visited.copy(),
                    'processed': processed_nodes.copy(),
                    'checking_node': neighbor,
                    'previous': self.previous.copy(),
                    'message': f'检查邻居 {neighbor}'
                })
                
                # 计算新距离（每个格子距离为1）
                new_dist = current_dist + 1
                
                if neighbor not in self.distances or new_dist < self.distances[neighbor]:
                    old_dist = self.distances.get(neighbor, float('inf'))
                    self.distances[neighbor] = new_dist
                    self.previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))
                    
                    # 记录更新距离
                    self.steps.append({
                        'distances': self.distances.copy(),
                        'current_node': current_node,
                        'visited': self.visited.copy(),
                        'processed': processed_nodes.copy(),
                        'updated_node': neighbor,
                        'old_distance': old_dist,
                        'new_distance': new_dist,
                        'previous': self.previous.copy(),
                        'message': f'更新节点 {neighbor}：{old_dist if old_dist == float("inf") else old_dist} → {new_dist}'
                    })
        
        # 记录完成状态
        if self.end in self.distances:
            path = self.get_path()
            self.steps.append({
                'distances': self.distances.copy(),
                'current_node': None,
                'visited': self.visited.copy(),
                'processed': processed_nodes.copy(),
                'previous': self.previous.copy(),
                'path': path,
                'message': f'算法完成！找到最短路径，长度：{len(path) - 1}'
            })
        else:
            self.steps.append({
                'distances': self.distances.copy(),
                'current_node': None,
                'visited': self.visited.copy(),
                'processed': processed_nodes.copy(),
                'previous': self.previous.copy(),
                'message': '算法完成！未找到路径'
            })
    
    def get_path(self):
        """获取从起点到终点的最短路径"""
        if self.end not in self.previous:
            return []
        
        path = []
        current = self.end
        while current is not None:
            path.append(current)
            current = self.previous.get(current)
        
        return path[::-1]  # 反转路径
    
    def expand_steps(self):
        """扩展步骤以增加帧数和流畅度"""
        expanded_steps = []
        target_frames = 300
        
        for i, step in enumerate(self.steps):
            # 重要步骤分配更多帧
            if step.get('current_node') is not None or step.get('updated_node') is not None:
                frames_per_step = 3
            else:
                frames_per_step = 2
            
            # 添加重复帧
            for _ in range(frames_per_step):
                expanded_steps.append(step.copy())
            
            # 在步骤之间添加过渡帧
            if i < len(self.steps) - 1:
                next_step = self.steps[i + 1]
                for transition in range(2):
                    transition_step = step.copy()
                    transition_step['transition_alpha'] = (transition + 1) / 3.0
                    transition_step['next_step'] = next_step
                    expanded_steps.append(transition_step)
        
        # 调整到目标帧数
        if len(expanded_steps) < target_frames:
            while len(expanded_steps) < target_frames:
                expanded_steps.append(self.steps[-1].copy())
        elif len(expanded_steps) > target_frames:
            indices = np.linspace(0, len(expanded_steps) - 1, target_frames, dtype=int)
            expanded_steps = [expanded_steps[i] for i in indices]
        
        self.steps = expanded_steps[:target_frames]
    
    def draw_maze(self, step_data):
        """绘制当前步骤的迷宫"""
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        distances = step_data.get('distances', {})
        current_node = step_data.get('current_node')
        visited = step_data.get('visited', set())
        processed = step_data.get('processed', set())
        checking_node = step_data.get('checking_node')
        updated_node = step_data.get('updated_node')
        previous = step_data.get('previous', {})
        path = step_data.get('path', [])
        
        # 创建颜色矩阵
        colors = np.zeros((self.height, self.width, 3))
        
        for i in range(self.height):
            for j in range(self.width):
                pos = (i, j)
                
                if self.maze[i, j] == 1:
                    # 墙壁：黑色
                    colors[i, j] = [0.2, 0.2, 0.2]
                elif pos == self.start:
                    # 起点：绿色
                    colors[i, j] = [0, 1, 0]
                elif pos == self.end:
                    # 终点：红色
                    colors[i, j] = [1, 0, 0]
                elif pos in path:
                    # 最短路径：黄色
                    colors[i, j] = [1, 1, 0]
                elif pos == current_node:
                    # 当前处理的节点：蓝色
                    colors[i, j] = [0, 0.5, 1]
                elif pos == checking_node:
                    # 正在检查的节点：浅蓝色
                    colors[i, j] = [0.5, 0.8, 1]
                elif pos == updated_node:
                    # 刚更新的节点：橙色
                    colors[i, j] = [1, 0.6, 0]
                elif pos in processed:
                    # 已处理的节点：浅绿色
                    colors[i, j] = [0.5, 1, 0.5]
                elif pos in visited:
                    # 已访问的节点：浅灰色
                    colors[i, j] = [0.8, 0.8, 0.8]
                else:
                    # 未访问的节点：白色
                    colors[i, j] = [1, 1, 1]
        
        # 绘制迷宫
        self.ax.imshow(colors, interpolation='nearest', origin='upper')
        
        # 添加距离标签（在已访问的节点上）
        for pos in visited:
            if pos in distances and distances[pos] != float('inf'):
                i, j = pos
                dist = distances[pos]
                self.ax.text(j, i, str(dist), fontsize=8, ha='center', va='center',
                           color='black', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                   edgecolor='gray', alpha=0.7))
        
        # 添加步骤信息
        message = step_data.get('message', '')
        self.ax.text(0.5, -0.5, message, fontsize=12, ha='left', va='top',
                    transform=self.ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                            edgecolor='black', alpha=0.9, linewidth=2),
                    fontweight='bold')
        
        # 添加图例
        legend_text = "图例：\n绿色=起点 | 红色=终点 | 蓝色=当前节点 | 浅蓝=检查中 | 橙色=刚更新\n浅绿=已处理 | 浅灰=已访问 | 黄色=最短路径 | 黑色=墙壁"
        self.ax.text(1.02, 0.98, legend_text, fontsize=10, ha='left', va='top',
                    transform=self.ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='gray', alpha=0.8, linewidth=1.5))
    
    def animate(self, frame):
        """动画函数"""
        if frame < len(self.steps):
            self.draw_maze(self.steps[frame])
        return []
    
    def create_animation(self, output_file=None, interval=100, dpi=150):
        """创建动画并保存为GIF"""
        # 设置默认输出文件路径
        if output_file is None:
            output_file = os.path.join(IMAGES_DIR, 'maze_dijkstra.gif')
        elif not os.path.isabs(output_file):
            # 如果是相对路径，则保存到images文件夹
            output_file = os.path.join(IMAGES_DIR, output_file)
        
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        
        total_frames = len(self.steps)
        print(f"正在生成迷宫Dijkstra动画，共 {total_frames} 帧...")
        print(f"这可能需要一些时间，请稍候...")
        
        anim = animation.FuncAnimation(self.fig, self.animate, 
                                      frames=total_frames,
                                      interval=interval, 
                                      repeat=True, 
                                      blit=False)
        
        fps = 1000 / interval if interval > 0 else 10
        anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
        
        print(f"\n✓ 动画已成功保存为: {output_file}")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 分辨率: {dpi} DPI")
        print(f"  - 每帧间隔: {interval}ms")
        print(f"  - 迷宫大小: {self.height} x {self.width}")
        if self.end in self.distances:
            path = self.get_path()
            print(f"  - 最短路径长度: {len(path) - 1}")
        
        return anim

if __name__ == '__main__':
    print("=" * 60)
    print("迷宫Dijkstra算法动态演示生成器")
    print("=" * 60)
    
    # 创建迷宫和可视化对象
    print("\n正在生成复杂迷宫...")
    maze_viz = MazeDijkstra(width=30, height=30, complexity=0.8, density=0.7)
    
    # 生成动画
    print("\n开始生成动画...")
    anim = maze_viz.create_animation(None, interval=80, dpi=150)  # None表示使用默认路径
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)

