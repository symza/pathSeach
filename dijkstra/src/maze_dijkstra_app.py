import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import heapq
from collections import deque

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PathfindingMazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("路径搜索算法演示 - Dijkstra & A*")
        self.root.geometry("1200x800")
        
        # 迷宫参数
        self.size = 40  # 正方形迷宫的边长
        self.width = 40
        self.height = 40
        self.complexity = 0.5
        self.density = 0.5
        
        # 迷宫和算法状态
        self.maze = None
        self.start = None
        self.end = None
        self.distances = {}
        self.previous = {}
        self.visited = set()
        self.path = []
        self.steps = []
        self.current_step = 0
        self.is_running = False
        self.is_paused = False
        self.speed = 5  # 毫秒（默认更快）
        self.algorithm = 'Dijkstra'  # 当前使用的算法
        
        # 创建界面
        self.create_widgets()
        
        # 初始化迷宫
        self.generate_maze()
    
    def update_complexity_label(self):
        """更新复杂度标签显示"""
        value = round(self.complexity_var.get(), 2)
        self.complexity_label.config(text="{:.2f}".format(value))
    
    def update_density_label(self):
        """更新密度标签显示"""
        value = round(self.density_var.get(), 2)
        self.density_label.config(text="{:.2f}".format(value))
    
    def on_algorithm_change(self):
        """当算法改变时更新"""
        self.algorithm = self.algorithm_var.get()
        if self.maze is not None:
            self.update_info("算法已切换为: {}".format(self.algorithm))
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 迷宫参数
        params_frame = ttk.LabelFrame(control_frame, text="迷宫参数", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 迷宫大小（正方形）
        ttk.Label(params_frame, text="大小:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.size_var = tk.IntVar(value=self.size)
        size_scale = ttk.Scale(params_frame, from_=20, to=60, variable=self.size_var, 
                              orient=tk.HORIZONTAL, length=200,
                              command=lambda v: self.size_var.set(int(float(v))))
        size_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.size_label = ttk.Label(params_frame, textvariable=self.size_var)
        self.size_label.grid(row=0, column=2, padx=5)
        
        # 复杂度
        ttk.Label(params_frame, text="复杂度:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.complexity_var = tk.DoubleVar(value=self.complexity)
        complexity_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.complexity_var, 
                                     orient=tk.HORIZONTAL, length=200,
                                     command=lambda v: self.complexity_var.set(round(float(v), 2)))
        complexity_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        self.complexity_label = ttk.Label(params_frame)
        self.complexity_label.grid(row=2, column=2, padx=5)
        self.update_complexity_label()
        self.complexity_var.trace('w', lambda *args: self.update_complexity_label())
        
        # 密度
        ttk.Label(params_frame, text="密度:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.density_var = tk.DoubleVar(value=self.density)
        density_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.density_var, 
                                 orient=tk.HORIZONTAL, length=200,
                                 command=lambda v: self.density_var.set(round(float(v), 2)))
        density_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5)
        self.density_label = ttk.Label(params_frame)
        self.density_label.grid(row=3, column=2, padx=5)
        self.update_density_label()
        self.density_var.trace('w', lambda *args: self.update_density_label())
        
        # 算法选择
        algorithm_frame = ttk.LabelFrame(control_frame, text="算法选择", padding="10")
        algorithm_frame.pack(fill=tk.X, pady=10)
        
        self.algorithm_var = tk.StringVar(value='Dijkstra')
        ttk.Radiobutton(algorithm_frame, text="Dijkstra算法", variable=self.algorithm_var, 
                       value='Dijkstra', command=self.on_algorithm_change).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(algorithm_frame, text="A*算法", variable=self.algorithm_var, 
                       value='A*', command=self.on_algorithm_change).pack(anchor=tk.W, pady=2)
        
        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="生成新迷宫", command=self.generate_maze).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="开始搜索", command=self.start_search).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="暂停/继续", command=self.pause_resume).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="重置", command=self.reset).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="显示路径", command=self.show_path).pack(fill=tk.X, pady=2)
        
        # 速度控制
        speed_frame = ttk.LabelFrame(control_frame, text="动画速度", padding="10")
        speed_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(speed_frame, text="速度 (越小越快):").pack()
        self.speed_var = tk.IntVar(value=self.speed)
        speed_scale = ttk.Scale(speed_frame, from_=1, to=50, variable=self.speed_var, 
                               orient=tk.HORIZONTAL, length=200)
        speed_scale.pack(fill=tk.X, pady=5)
        ttk.Label(speed_frame, textvariable=self.speed_var).pack()
        
        # 信息显示
        info_frame = ttk.LabelFrame(control_frame, text="信息", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.info_text = tk.Text(info_frame, height=10, width=30, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # 右侧图形显示
        fig_frame = ttk.Frame(main_frame)
        fig_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def generate_maze(self):
        """生成新迷宫"""
        if self.is_running:
            messagebox.showwarning("警告", "算法正在运行，请先停止")
            return
        
        # 更新参数
        self.size = self.size_var.get()
        self.width = self.size
        self.height = self.size
        self.complexity = self.complexity_var.get()
        self.density = self.density_var.get()
        
        # 使用递归分割算法生成迷宫
        self.maze = self.generate_maze_recursive_division()
        
        # 随机选择起点和终点
        self.find_start_end()
        
        # 确保起点和终点之间有通路
        if not self.has_path(self.start, self.end):
            # 如果起点和终点不连通，创建一条路径
            self.create_path_between(self.start, self.end)
        
        # 重置算法状态
        self.reset()
        
        # 绘制迷宫
        self.draw_maze()
        
        self.update_info("迷宫已生成\n起点: {}\n终点: {}".format(self.start, self.end))
    
    def generate_maze_recursive_division(self):
        """使用递归分割算法生成迷宫"""
        # 初始化：全部是通路
        maze = np.zeros((self.height, self.width), dtype=int)
        
        # 设置边界为单层墙壁
        for j in range(self.width):
            maze[0, j] = 1
            maze[self.height - 1, j] = 1
        for i in range(self.height):
            maze[i, 0] = 1
            maze[i, self.width - 1] = 1
        
        # 递归分割生成迷宫（从边界内侧开始，确保边界内侧也有结构）
        self._divide(maze, 1, 1, self.height - 2, self.width - 2)
        
        # 根据复杂度参数添加少量的额外通路
        num_extra_paths = int(self.complexity * (self.height * self.width) * 0.03)
        attempts = 0
        max_attempts = num_extra_paths * 10
        while attempts < max_attempts and num_extra_paths > 0:
            row = random.randint(2, self.height - 3)
            col = random.randint(2, self.width - 3)
            if maze[row, col] == 1:  # 如果是墙壁
                adjacent_paths = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                    if 0 <= row + dr < self.height and 0 <= col + dc < self.width
                                    and maze[row + dr, col + dc] == 0)
                if adjacent_paths >= 3:
                    maze[row, col] = 0
                    num_extra_paths -= 1
            attempts += 1
        
        # 根据密度参数添加少量死胡同
        num_dead_ends = int((1 - self.density) * (self.height * self.width) * 0.05)
        attempts = 0
        max_attempts = num_dead_ends * 10
        while attempts < max_attempts and num_dead_ends > 0:
            row = random.randint(2, self.height - 3)
            col = random.randint(2, self.width - 3)
            if maze[row, col] == 0:  # 如果是通路
                adjacent_walls = [(dr, dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= row + dr < self.height and 0 <= col + dc < self.width
                                and maze[row + dr, col + dc] == 1]
                if len(adjacent_walls) >= 3:
                    dr, dc = random.choice(adjacent_walls)
                    nr, nc = row + dr, col + dc
                    if 2 <= nr < self.height - 2 and 2 <= nc < self.width - 2:
                        length = random.randint(1, 2)
                        for i in range(length):
                            tr, tc = row + dr * (i + 1), col + dc * (i + 1)
                            if 2 <= tr < self.height - 2 and 2 <= tc < self.width - 2:
                                maze[tr, tc] = 0
                            else:
                                break
                        num_dead_ends -= 1
            attempts += 1
        
        return maze
    
    def _divide(self, maze, row_start, col_start, row_end, col_end):
        """递归分割区域生成迷宫"""
        # 如果区域太小，停止分割
        if row_end - row_start < 2 or col_end - col_start < 2:
            return
        
        # 随机选择分割方向
        if row_end - row_start == col_end - col_start:
            split_horizontal = random.random() < 0.5
        else:
            split_horizontal = row_end - row_start > col_end - col_start
        
        if split_horizontal:
            # 水平分割
            # 确保墙壁位置在有效范围内
            if row_end - row_start < 2:
                return
            wall_row = random.randint(row_start + 1, row_end - 1)
            # 创建墙壁
            for j in range(col_start, col_end + 1):
                maze[wall_row, j] = 1
            # 在墙壁上随机开一个洞
            hole = random.randint(col_start, col_end)
            maze[wall_row, hole] = 0
            # 递归分割
            self._divide(maze, row_start, col_start, wall_row - 1, col_end)
            self._divide(maze, wall_row + 1, col_start, row_end, col_end)
        else:
            # 垂直分割
            # 确保墙壁位置在有效范围内
            if col_end - col_start < 2:
                return
            wall_col = random.randint(col_start + 1, col_end - 1)
            # 创建墙壁
            for i in range(row_start, row_end + 1):
                maze[i, wall_col] = 1
            # 在墙壁上随机开一个洞
            hole = random.randint(row_start, row_end)
            maze[hole, wall_col] = 0
            # 递归分割
            self._divide(maze, row_start, col_start, row_end, wall_col - 1)
            self._divide(maze, row_start, wall_col + 1, row_end, col_end)
    
    def find_start_end(self):
        """随机选择起点和终点"""
        # 找到所有通路位置
        path_cells = []
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                if self.maze[i, j] == 0:
                    path_cells.append((i, j))
        
        if len(path_cells) < 2:
            self.start = (1, 1)
            self.end = (self.height - 2, self.width - 2)
            return
        
        # 随机选择起点和终点，确保它们足够远
        max_attempts = 100
        for _ in range(max_attempts):
            self.start = random.choice(path_cells)
            self.end = random.choice(path_cells)
            if self.start != self.end:
                # 计算距离
                dist = abs(self.start[0] - self.end[0]) + abs(self.start[1] - self.end[1])
                if dist >= min(self.height, self.width) // 3:
                    # 确保起点和终点周围是通路
                    if self.is_valid_cell(self.start) and self.is_valid_cell(self.end):
                        return
        
        # 如果找不到合适的点，使用默认值
        self.start = path_cells[0]
        self.end = path_cells[-1]
    
    def is_valid_cell(self, cell):
        """检查单元格是否有效（是通路且不在边界）"""
        row, col = cell
        return (1 <= row < self.height - 1 and 1 <= col < self.width - 1 
                and self.maze[row, col] == 0)
    
    def has_path(self, start, end):
        """使用BFS检查起点和终点之间是否有通路"""
        if start == end:
            return True
        
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            
            if current == end:
                return True
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if neighbor in visited:
                    continue
                
                if not self.is_valid_cell(neighbor):
                    continue
                
                visited.add(neighbor)
                queue.append(neighbor)
        
        return False
    
    def create_path_between(self, start, end):
        """在起点和终点之间创建一条路径"""
        # 使用简单的曼哈顿路径（先横向再纵向，或先纵向再横向）
        # 随机选择路径方向
        if random.random() < 0.5:
            # 先横向再纵向
            # 横向路径
            row_start, col_start = start
            row_end, col_end = end
            
            # 从起点横向移动到终点列
            for col in range(min(col_start, col_end), max(col_start, col_end) + 1):
                if 1 <= row_start < self.height - 1 and 1 <= col < self.width - 1:
                    self.maze[row_start, col] = 0
            
            # 从起点行纵向移动到终点行
            for row in range(min(row_start, row_end), max(row_start, row_end) + 1):
                if 1 <= row < self.height - 1 and 1 <= col_end < self.width - 1:
                    self.maze[row, col_end] = 0
        else:
            # 先纵向再横向
            row_start, col_start = start
            row_end, col_end = end
            
            # 从起点纵向移动到终点行
            for row in range(min(row_start, row_end), max(row_start, row_end) + 1):
                if 1 <= row < self.height - 1 and 1 <= col_start < self.width - 1:
                    self.maze[row, col_start] = 0
            
            # 从起点列横向移动到终点列
            for col in range(min(col_start, col_end), max(col_start, col_end) + 1):
                if 1 <= row_end < self.height - 1 and 1 <= col < self.width - 1:
                    self.maze[row_end, col] = 0
    
    def reset(self):
        """重置算法状态"""
        self.distances = {}
        self.previous = {}
        self.visited = set()
        self.path = []
        self.steps = []
        self.current_step = 0
        self.is_running = False
        self.is_paused = False
        self.draw_maze()
        self.update_info("已重置")
    
    def start_search(self):
        """开始Dijkstra搜索"""
        if self.maze is None:
            messagebox.showwarning("警告", "请先生成迷宫")
            return
        
        if self.is_running and not self.is_paused:
            return
        
        if self.is_paused:
            self.is_paused = False
            self.continue_search()
            return
        
        # 初始化
        self.distances = {self.start: 0}
        self.previous = {}
        self.visited = set()
        self.path = []
        self.steps = []
        self.current_step = 0
        
        # 运行算法
        self.algorithm = self.algorithm_var.get()
        if self.algorithm == 'A*':
            self.run_astar()
        else:
            self.run_dijkstra()
    
    def run_dijkstra(self):
        """运行Dijkstra算法"""
        self.is_running = True
        
        # 优先队列：(距离, 位置)
        pq = [(0, self.start)]
        
        while pq:
            if self.is_paused:
                return
            
            current_dist, current = heapq.heappop(pq)
            
            if current in self.visited:
                continue
            
            self.visited.add(current)
            
            # 记录步骤
            self.steps.append({
                'current': current,
                'visited': self.visited.copy(),
                'distances': self.distances.copy()
            })
            
            # 如果到达终点，停止
            if current == self.end:
                break
            
            # 检查邻居
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self.is_valid_cell(neighbor):
                    continue
                
                if neighbor in self.visited:
                    continue
                
                # 计算新距离（每个移动的代价为1）
                new_dist = current_dist + 1
                
                if neighbor not in self.distances or new_dist < self.distances[neighbor]:
                    self.distances[neighbor] = new_dist
                    self.previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        # 重建路径
        self.reconstruct_path()
        
        # 开始动画
        self.animate_search()
    
    def run_astar(self):
        """运行A*算法"""
        self.is_running = True
        
        # 优先队列：(f值, g值, 位置)
        # f(n) = g(n) + h(n)，其中g(n)是实际距离，h(n)是启发式估计
        pq = [(self.heuristic(self.start), 0, self.start)]
        
        while pq:
            if self.is_paused:
                return
            
            f_value, current_dist, current = heapq.heappop(pq)
            
            if current in self.visited:
                continue
            
            self.visited.add(current)
            
            # 记录步骤
            self.steps.append({
                'current': current,
                'visited': self.visited.copy(),
                'distances': self.distances.copy()
            })
            
            # 如果到达终点，停止
            if current == self.end:
                break
            
            # 检查邻居
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self.is_valid_cell(neighbor):
                    continue
                
                if neighbor in self.visited:
                    continue
                
                # 计算新距离（每个移动的代价为1）
                new_dist = current_dist + 1
                
                if neighbor not in self.distances or new_dist < self.distances[neighbor]:
                    self.distances[neighbor] = new_dist
                    self.previous[neighbor] = current
                    # f值 = g值 + h值
                    f_value = new_dist + self.heuristic(neighbor)
                    heapq.heappush(pq, (f_value, new_dist, neighbor))
        
        # 重建路径
        self.reconstruct_path()
        
        # 开始动画
        self.animate_search()
    
    def reconstruct_path(self):
        """重建最短路径"""
        if self.end not in self.previous:
            self.path = []
            return
        
        self.path = []
        current = self.end
        while current is not None:
            self.path.append(current)
            current = self.previous.get(current)
        
        self.path.reverse()
    
    def pause_resume(self):
        """暂停/继续"""
        if not self.is_running:
            return
        
        self.is_paused = not self.is_paused
        
        if not self.is_paused:
            self.continue_search()
    
    def continue_search(self):
        """继续搜索动画"""
        if self.current_step < len(self.steps):
            self.animate_search()
        else:
            # 如果已经完成，显示最终结果
            self.draw_final_result()
    
    def animate_search(self):
        """动画显示搜索过程"""
        if self.current_step >= len(self.steps):
            self.is_running = False
            self.draw_final_result()
            return
        
        if self.is_paused:
            return
        
        step = self.steps[self.current_step]
        self.draw_search_step(step)
        
        self.current_step += 1
        
        # 继续下一帧
        self.root.after(self.speed_var.get(), self.animate_search)
    
    def draw_search_step(self, step):
        """绘制搜索步骤"""
        self.ax.clear()
        
        # 绘制迷宫
        self.ax.imshow(self.maze, cmap='gray_r', interpolation='nearest')
        
        # 绘制已访问的节点（浅蓝色）
        for cell in step['visited']:
            if cell != step['current']:
                self.ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, 
                                               facecolor='lightblue', alpha=0.5))
        
        # 绘制当前节点（橙色）
        if step['current']:
            self.ax.add_patch(plt.Rectangle((step['current'][1] - 0.5, step['current'][0] - 0.5), 
                                           1, 1, facecolor='orange', alpha=0.8))
        
        # 绘制起点（绿色）
        if self.start:
            self.ax.add_patch(plt.Rectangle((self.start[1] - 0.5, self.start[0] - 0.5), 
                                           1, 1, facecolor='green', alpha=0.8))
        
        # 绘制终点（红色）
        if self.end:
            self.ax.add_patch(plt.Rectangle((self.end[1] - 0.5, self.end[0] - 0.5), 
                                           1, 1, facecolor='red', alpha=0.8))
        
        self.ax.set_xlim(-0.5, self.width - 0.5)
        self.ax.set_ylim(self.height - 0.5, -0.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        algorithm_name = self.algorithm if hasattr(self, 'algorithm') else 'Dijkstra'
        self.ax.set_title('{}算法搜索中... (步骤 {}/{})'.format(
            algorithm_name, self.current_step + 1, len(self.steps)), fontsize=12, fontweight='bold')
        
        self.canvas.draw()
        
        # 更新信息
        if step['current']:
            dist = step['distances'].get(step['current'], float('inf'))
            self.update_info("当前节点: {}\n距离: {}\n已访问: {}".format(
                step['current'], dist, len(step['visited'])))
    
    def draw_final_result(self):
        """绘制最终结果"""
        self.ax.clear()
        
        # 绘制迷宫
        self.ax.imshow(self.maze, cmap='gray_r', interpolation='nearest')
        
        # 绘制已访问的节点（浅蓝色）
        for cell in self.visited:
            if cell not in self.path:
                self.ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, 
                                             facecolor='lightblue', alpha=0.5))
        
        # 绘制最短路径（黄色）
        for cell in self.path:
            self.ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, 
                                           facecolor='yellow', alpha=0.8, edgecolor='orange', linewidth=2))
        
        # 绘制起点（绿色）
        if self.start:
            self.ax.add_patch(plt.Rectangle((self.start[1] - 0.5, self.start[0] - 0.5), 
                                           1, 1, facecolor='green', alpha=0.8))
        
        # 绘制终点（红色）
        if self.end:
            self.ax.add_patch(plt.Rectangle((self.end[1] - 0.5, self.end[0] - 0.5), 
                                           1, 1, facecolor='red', alpha=0.8))
        
        self.ax.set_xlim(-0.5, self.width - 0.5)
        self.ax.set_ylim(self.height - 0.5, -0.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        if self.path:
            path_length = len(self.path) - 1
            self.ax.set_title('搜索完成！最短路径长度: {}'.format(path_length), 
                           fontsize=12, fontweight='bold', color='green')
            self.update_info("搜索完成！\n最短路径长度: {}\n已访问节点数: {}".format(
                path_length, len(self.visited)))
        else:
            self.ax.set_title('未找到路径', fontsize=12, fontweight='bold', color='red')
            self.update_info("未找到从起点到终点的路径")
        
        self.canvas.draw()
        self.is_running = False
    
    def show_path(self):
        """显示路径"""
        if not self.path:
            messagebox.showinfo("提示", "请先运行搜索算法")
            return
        
        self.draw_final_result()
    
    def draw_maze(self):
        """绘制迷宫"""
        if self.maze is None:
            return
        
        self.ax.clear()
        self.ax.imshow(self.maze, cmap='gray_r', interpolation='nearest')
        
        # 绘制起点和终点
        if self.start:
            self.ax.add_patch(plt.Rectangle((self.start[1] - 0.5, self.start[0] - 0.5), 
                                           1, 1, facecolor='green', alpha=0.8))
        if self.end:
            self.ax.add_patch(plt.Rectangle((self.end[1] - 0.5, self.end[0] - 0.5), 
                                           1, 1, facecolor='red', alpha=0.8))
        
        self.ax.set_xlim(-0.5, self.width - 0.5)
        self.ax.set_ylim(self.height - 0.5, -0.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title('迷宫 (绿色=起点, 红色=终点)', fontsize=12, fontweight='bold')
        
        self.canvas.draw()
    
    def update_info(self, text):
        """更新信息显示"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingMazeApp(root)
    root.mainloop()

