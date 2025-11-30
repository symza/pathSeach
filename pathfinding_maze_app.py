import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import heapq
from collections import deque
import os
from PIL import Image
import io

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PathfindingMazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("路径搜索算法演示 - Dijkstra、A*、RRT")
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
        
        # 算法参数
        # A*算法参数
        self.astar_heuristic_weight = 1.0  # 启发式权重
        
        # RRT算法参数
        self.rrt_step_size_factor = 6.0  # 步长因子（步长 = 迷宫尺寸 / 因子）
        self.rrt_goal_bias = 0.3  # 目标偏向度（0-1，越接近1越偏向目标）
        self.rrt_max_iterations = 5000  # 最大迭代次数
        
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
        self.update_algorithm_params_visibility()
        if self.maze is not None:
            self.update_info("算法已切换为: {}".format(self.algorithm))
    
    def update_algorithm_params_visibility(self):
        """根据选择的算法显示/隐藏参数面板"""
        # 隐藏所有参数面板
        for widget in self.algorithm_params_frame.winfo_children():
            widget.pack_forget()
        
        # 获取当前选择的算法
        current_algorithm = self.algorithm_var.get() if hasattr(self, 'algorithm_var') else self.algorithm
        
        # 显示当前算法的参数面板
        if current_algorithm == 'A*':
            self.astar_params_frame.pack(fill=tk.X)
        elif current_algorithm == 'RRT':
            self.rrt_params_frame.pack(fill=tk.X)
    
    def create_astar_params(self):
        """创建A*算法参数面板"""
        self.astar_params_frame = ttk.Frame(self.algorithm_params_frame)
        
        # 启发式权重
        ttk.Label(self.astar_params_frame, text="启发式权重:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.astar_weight_var = tk.DoubleVar(value=self.astar_heuristic_weight)
        weight_scale = ttk.Scale(self.astar_params_frame, from_=0.1, to=5.0, 
                                variable=self.astar_weight_var, orient=tk.HORIZONTAL, length=200,
                                command=lambda v: self.astar_weight_var.set(round(float(v), 2)))
        weight_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.astar_weight_label = ttk.Label(self.astar_params_frame, text="1.0")
        self.astar_weight_label.grid(row=0, column=2, padx=5)
        
        # 更新标签显示
        def update_astar_weight_label(*args):
            value = round(self.astar_weight_var.get(), 2)
            self.astar_weight_label.config(text="{:.2f}".format(value))
            self.astar_heuristic_weight = value
        
        self.astar_weight_var.trace('w', lambda *args: update_astar_weight_label())
        update_astar_weight_label()
        
        # 说明文本
        info_label = ttk.Label(self.astar_params_frame, 
                              text="权重越大，算法越倾向于向目标移动（更快但可能不是最优）",
                              font=('Arial', 8), foreground='gray')
        info_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
    
    def create_rrt_params(self):
        """创建RRT算法参数面板"""
        self.rrt_params_frame = ttk.Frame(self.algorithm_params_frame)
        
        # 步长因子
        ttk.Label(self.rrt_params_frame, text="步长因子:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.rrt_step_size_var = tk.DoubleVar(value=self.rrt_step_size_factor)
        step_scale = ttk.Scale(self.rrt_params_frame, from_=3.0, to=15.0, 
                              variable=self.rrt_step_size_var, orient=tk.HORIZONTAL, length=200,
                              command=lambda v: self.rrt_step_size_var.set(round(float(v), 1)))
        step_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.rrt_step_size_label = ttk.Label(self.rrt_params_frame, text="6.0")
        self.rrt_step_size_label.grid(row=0, column=2, padx=5)
        
        # 目标偏向度
        ttk.Label(self.rrt_params_frame, text="目标偏向度:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.rrt_goal_bias_var = tk.DoubleVar(value=self.rrt_goal_bias)
        bias_scale = ttk.Scale(self.rrt_params_frame, from_=0.0, to=1.0, 
                              variable=self.rrt_goal_bias_var, orient=tk.HORIZONTAL, length=200,
                              command=lambda v: self.rrt_goal_bias_var.set(round(float(v), 2)))
        bias_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        self.rrt_goal_bias_label = ttk.Label(self.rrt_params_frame, text="0.30")
        self.rrt_goal_bias_label.grid(row=1, column=2, padx=5)
        
        # 最大迭代次数
        ttk.Label(self.rrt_params_frame, text="最大迭代次数:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.rrt_max_iter_var = tk.IntVar(value=self.rrt_max_iterations)
        iter_scale = ttk.Scale(self.rrt_params_frame, from_=1000, to=10000, 
                              variable=self.rrt_max_iter_var, orient=tk.HORIZONTAL, length=200,
                              command=lambda v: self.rrt_max_iter_var.set(int(float(v))))
        iter_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        self.rrt_max_iter_label = ttk.Label(self.rrt_params_frame, text="5000")
        self.rrt_max_iter_label.grid(row=2, column=2, padx=5)
        
        # 更新标签显示
        def update_rrt_step_label(*args):
            value = round(self.rrt_step_size_var.get(), 1)
            self.rrt_step_size_label.config(text="{:.1f}".format(value))
            self.rrt_step_size_factor = value
        
        def update_rrt_bias_label(*args):
            value = round(self.rrt_goal_bias_var.get(), 2)
            self.rrt_goal_bias_label.config(text="{:.2f}".format(value))
            self.rrt_goal_bias = value
        
        def update_rrt_iter_label(*args):
            value = self.rrt_max_iter_var.get()
            self.rrt_max_iter_label.config(text="{}".format(value))
            self.rrt_max_iterations = value
        
        self.rrt_step_size_var.trace('w', lambda *args: update_rrt_step_label())
        self.rrt_goal_bias_var.trace('w', lambda *args: update_rrt_bias_label())
        self.rrt_max_iter_var.trace('w', lambda *args: update_rrt_iter_label())
        
        update_rrt_step_label()
        update_rrt_bias_label()
        update_rrt_iter_label()
        
        # 说明文本
        info_label = ttk.Label(self.rrt_params_frame, 
                              text="步长因子越小步长越大；偏向度越高越倾向目标；迭代次数越多成功率越高",
                              font=('Arial', 8), foreground='gray', wraplength=250)
        info_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
    
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
        ttk.Radiobutton(algorithm_frame, text="RRT算法", variable=self.algorithm_var, 
                       value='RRT', command=self.on_algorithm_change).pack(anchor=tk.W, pady=2)
        
        # 算法参数框架
        self.algorithm_params_frame = ttk.LabelFrame(control_frame, text="算法参数", padding="10")
        self.algorithm_params_frame.pack(fill=tk.X, pady=10)
        
        # 创建A*参数面板
        self.create_astar_params()
        
        # 创建RRT参数面板
        self.create_rrt_params()
        
        # 初始时隐藏参数面板
        self.update_algorithm_params_visibility()
        
        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="生成新迷宫", command=self.generate_maze).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="开始搜索", command=self.start_search).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="暂停/继续", command=self.pause_resume).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="重置", command=self.reset).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="显示路径", command=self.show_path).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="生成动图", command=self.generate_gif).pack(fill=tk.X, pady=2)
        
        # 速度控制
        speed_frame = ttk.LabelFrame(control_frame, text="动画速度", padding="10")
        speed_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(speed_frame, text="速度 (越小越快):").pack()
        self.speed_var = tk.IntVar(value=self.speed)
        speed_scale = ttk.Scale(speed_frame, from_=1, to=50, variable=self.speed_var, 
                               orient=tk.HORIZONTAL, length=200,
                               command=lambda v: self.speed_var.set(int(float(v))))
        speed_scale.pack(fill=tk.X, pady=5)
        self.speed_label = ttk.Label(speed_frame, textvariable=self.speed_var)
        self.speed_label.pack()
        
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
        
        # 初始化算法参数面板的显示
        self.update_algorithm_params_visibility()
    
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
        """开始路径搜索"""
        if self.maze is None:
            messagebox.showwarning("警告", "请先生成迷宫")
            return
        
        if self.start is None or self.end is None:
            messagebox.showwarning("警告", "起点或终点未设置")
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
        try:
            self.algorithm = self.algorithm_var.get()
        except AttributeError:
            self.algorithm = 'Dijkstra'
        
        self.update_info("开始运行{}算法...".format(self.algorithm))
        
        if self.algorithm == 'A*':
            self.run_astar()
        elif self.algorithm == 'RRT':
            self.run_rrt()
        else:
            self.run_dijkstra()
    
    def run_dijkstra(self):
        """运行Dijkstra算法"""
        try:
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
        except Exception as e:
            self.is_running = False
            messagebox.showerror("错误", "算法运行出错: {}".format(str(e)))
            import traceback
            traceback.print_exc()
    
    def heuristic(self, cell):
        """A*算法的启发式函数（曼哈顿距离）"""
        if self.end is None:
            return 0
        return abs(cell[0] - self.end[0]) + abs(cell[1] - self.end[1])
    
    def run_astar(self):
        """运行A*算法"""
        try:
            self.is_running = True
            
            # 初始化起点的g值
            self.distances[self.start] = 0
            
            # 优先队列：(f值, g值, 位置)
            # f(n) = g(n) + weight * h(n)，其中g(n)是实际距离，h(n)是启发式估计
            h_start = self.heuristic(self.start)
            f_start = 0 + self.astar_heuristic_weight * h_start
            pq = [(f_start, 0, self.start)]
            
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
                    
                    # 只有当找到更短的路径时才更新
                    if neighbor not in self.distances or new_dist < self.distances[neighbor]:
                        self.distances[neighbor] = new_dist
                        self.previous[neighbor] = current
                        # f值 = g值 + weight * h值（使用可配置的启发式权重）
                        h_value = self.heuristic(neighbor)
                        f_value = new_dist + self.astar_heuristic_weight * h_value
                        heapq.heappush(pq, (f_value, new_dist, neighbor))
            
            # 重建路径
            self.reconstruct_path()
            
            # 开始动画
            self.animate_search()
        except Exception as e:
            self.is_running = False
            messagebox.showerror("错误", "A*算法运行出错: {}".format(str(e)))
            import traceback
            traceback.print_exc()
    
    def run_rrt(self):
        """运行RRT（Rapidly-exploring Random Tree）算法"""
        try:
            self.is_running = True
            
            # RRT树结构：{节点: 父节点}
            tree = {self.start: None}
            self.previous = {self.start: None}
            self.distances = {self.start: 0}
            self.visited = set([self.start])
            
            # 步长（每次扩展的最大距离，使用可配置的步长因子）
            step_size = max(3, int(min(self.height, self.width) / self.rrt_step_size_factor))
            
            # 目标接近阈值
            goal_threshold = step_size * 2
            
            # 最大迭代次数（使用可配置的值）
            max_iterations = self.rrt_max_iterations
            
            # 预计算所有通路位置（避免每次都遍历）
            path_cells = []
            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    if self.maze[i, j] == 0:
                        path_cells.append((i, j))
            
            if not path_cells:
                messagebox.showwarning("警告", "迷宫中没有通路")
                self.is_running = False
                return
            
            iteration = 0
            steps_recorded = 0
            
            while iteration < max_iterations:
                if self.is_paused:
                    return
                
                iteration += 1
                
                # 每100次迭代更新一次界面，避免卡死
                if iteration % 100 == 0:
                    self.root.update()
                
                # 随机采样（使用可配置的目标偏向度）
                if random.random() < self.rrt_goal_bias:
                    sample = self.end
                else:
                    sample = random.choice(path_cells)
                
                # 找到树中离采样点最近的节点（优化：避免每次都计算所有距离）
                min_dist = float('inf')
                nearest = self.start
                for node in tree.keys():
                    dist_sq = (node[0] - sample[0])**2 + (node[1] - sample[1])**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        nearest = node
                
                # 计算方向向量
                dx = sample[0] - nearest[0]
                dy = sample[1] - nearest[1]
                dist = (dx**2 + dy**2)**0.5
                
                if dist < 0.1:  # 距离太近，跳过
                    continue
                
                # 如果距离小于步长，直接使用采样点
                if dist <= step_size:
                    new_node = sample
                else:
                    # 否则向采样点方向扩展步长距离
                    new_node = (
                        int(nearest[0] + dx * step_size / dist),
                        int(nearest[1] + dy * step_size / dist)
                    )
                
                # 检查新节点是否有效
                if not self.is_valid_cell(new_node):
                    continue
                
                # 检查从最近节点到新节点的路径是否穿过墙壁（简化检查）
                if not self.is_path_clear_simple(nearest, new_node):
                    continue
                
                # 将新节点加入树
                tree[new_node] = nearest
                self.previous[new_node] = nearest
                actual_dist = ((new_node[0] - nearest[0])**2 + (new_node[1] - nearest[1])**2)**0.5
                self.distances[new_node] = self.distances[nearest] + actual_dist
                self.visited.add(new_node)
                
                # 每5次成功扩展记录一次步骤（减少步骤数量但保持流畅）
                steps_recorded += 1
                if steps_recorded % 5 == 0:
                    self.steps.append({
                        'current': new_node,
                        'visited': self.visited.copy(),
                        'distances': self.distances.copy(),
                        'tree': dict(tree)  # 浅拷贝
                    })
                
                # 检查是否接近终点
                dist_to_goal = ((new_node[0] - self.end[0])**2 + 
                              (new_node[1] - self.end[1])**2)**0.5
                
                if dist_to_goal <= goal_threshold:
                    # 检查到终点的路径是否清晰
                    if self.is_path_clear_simple(new_node, self.end):
                        # 连接终点
                        self.previous[self.end] = new_node
                        self.distances[self.end] = self.distances[new_node] + dist_to_goal
                        self.visited.add(self.end)
                        
                        # 记录最终步骤
                        self.steps.append({
                            'current': self.end,
                            'visited': self.visited.copy(),
                            'distances': self.distances.copy(),
                            'tree': dict(tree)
                        })
                        break
            
            # 如果没找到路径，尝试连接树中离终点最近的节点
            if self.end not in self.previous and tree:
                self.update_info("正向RRT未找到路径，尝试连接最近节点...")
                # 找到树中离终点最近的节点
                closest_to_end = min(tree.keys(), 
                                   key=lambda n: ((n[0] - self.end[0])**2 + 
                                                 (n[1] - self.end[1])**2)**0.5)
                dist_to_end = ((closest_to_end[0] - self.end[0])**2 + 
                             (closest_to_end[1] - self.end[1])**2)**0.5
                # 如果距离在合理范围内，尝试连接
                if dist_to_end <= goal_threshold * 3:
                    if self.is_path_clear_simple(closest_to_end, self.end):
                        self.previous[self.end] = closest_to_end
                        self.distances[self.end] = self.distances[closest_to_end] + dist_to_end
                        self.visited.add(self.end)
            
            # 重建路径
            self.reconstruct_path()
            
            if not self.path:
                self.update_info("RRT算法未找到路径\n（这是概率算法的正常情况，可以重试）")
            
            # 开始动画
            self.animate_search()
        except Exception as e:
            self.is_running = False
            messagebox.showerror("错误", "RRT算法运行出错: {}".format(str(e)))
            import traceback
            traceback.print_exc()
    
    def is_path_clear_simple(self, start, end):
        """简化的路径检查（只检查起点、终点和中间几个点）"""
        x0, y0 = start
        x1, y1 = end
        
        # 检查起点和终点
        if not self.is_valid_cell(start) or not self.is_valid_cell(end):
            return False
        
        # 如果距离很近，直接返回True
        dist = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        if dist < 2:
            return True
        
        # 检查中间的几个点（简化版）
        num_checks = max(3, int(dist))
        for i in range(1, num_checks):
            t = i / num_checks
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            if not self.is_valid_cell((x, y)):
                return False
        
        return True
    
    def is_path_clear(self, start, end):
        """检查从起点到终点的路径是否清晰（不穿过墙壁）"""
        # 使用Bresenham算法检查路径上的所有点
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # 检查当前点是否是墙壁
            if not self.is_valid_cell((x, y)):
                return False
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def reconstruct_path(self):
        """重建最短路径"""
        if self.end not in self.previous:
            self.path = []
            return
        
        self.path = []
        current = self.end
        
        # 从终点回溯到起点
        visited_nodes = set()
        while current is not None:
            if current in visited_nodes:
                # 防止循环
                break
            visited_nodes.add(current)
            self.path.append(current)
            current = self.previous.get(current)
        
        self.path.reverse()
        
        # 对于RRT算法，确保路径是连续的（填充路径中的间隙）
        if hasattr(self, 'algorithm') and self.algorithm == 'RRT' and len(self.path) > 1:
            filled_path = [self.path[0]]
            for i in range(1, len(self.path)):
                start = filled_path[-1]
                end = self.path[i]
                # 如果两点不直接相邻，填充中间点
                if abs(start[0] - end[0]) > 1 or abs(start[1] - end[1]) > 1:
                    # 使用Bresenham算法填充路径
                    intermediate = self.get_line_points(start, end)
                    filled_path.extend(intermediate[1:])  # 跳过起点（已存在）
                else:
                    filled_path.append(end)
            self.path = filled_path
    
    def get_line_points(self, start, end):
        """使用Bresenham算法获取两点之间的所有点"""
        x0, y0 = start
        x1, y1 = end
        points = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
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
        
        # 绘制已访问的节点（蓝色）
        for cell in step['visited']:
            if cell != step['current']:
                self.ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, 
                                               facecolor='steelblue', alpha=0.7))
        
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
        
        # 如果是RRT算法，绘制树结构
        if algorithm_name == 'RRT' and 'tree' in step:
            # 绘制树的所有边
            for node, parent in step['tree'].items():
                if parent is not None:
                    self.ax.plot([parent[1], node[1]], [parent[0], node[0]], 
                              'b-', linewidth=1, alpha=0.3)
        
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
        
        # 绘制已访问的节点（蓝色）
        for cell in self.visited:
            if cell not in self.path:
                self.ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, 
                                             facecolor='steelblue', alpha=0.7))
        
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
    
    def expand_steps_to_500(self):
        """将步骤扩展到500帧"""
        if not self.steps:
            return
        
        target_frames = 500
        current_frames = len(self.steps)
        
        if current_frames >= target_frames:
            # 如果帧数已经够多，均匀采样到500帧
            indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
            expanded_steps = [self.steps[i].copy() for i in indices]
        else:
            # 如果帧数不够，扩展帧数
            expanded_steps = []
            step_ratio = target_frames / current_frames
            
            for i, step in enumerate(self.steps):
                # 每个原始步骤分配多帧
                frames_for_step = max(1, int(step_ratio))
                for _ in range(frames_for_step):
                    expanded_steps.append(step.copy())
            
            # 如果还不够，用最后一帧填充
            while len(expanded_steps) < target_frames:
                expanded_steps.append(self.steps[-1].copy())
            
            # 如果多了，截取前500帧
            expanded_steps = expanded_steps[:target_frames]
        
        self.steps = expanded_steps
    
    def draw_step_to_image(self, step, step_num, total_steps):
        """绘制步骤到图像并返回PIL Image"""
        # 创建临时figure用于生成图像
        fig, ax = plt.subplots(figsize=(10, 8), dpi=80)
        
        # 绘制迷宫
        ax.imshow(self.maze, cmap='gray_r', interpolation='nearest')
        
        # 绘制已访问的节点（蓝色）
        for cell in step['visited']:
            if cell != step.get('current'):
                ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, 
                                          facecolor='steelblue', alpha=0.7))
        
        # 绘制当前节点（橙色）
        if step.get('current'):
            ax.add_patch(plt.Rectangle((step['current'][1] - 0.5, step['current'][0] - 0.5), 
                                      1, 1, facecolor='orange', alpha=0.8))
        
        # 绘制起点（绿色）
        if self.start:
            ax.add_patch(plt.Rectangle((self.start[1] - 0.5, self.start[0] - 0.5), 
                                      1, 1, facecolor='green', alpha=0.8))
        
        # 绘制终点（红色）
        if self.end:
            ax.add_patch(plt.Rectangle((self.end[1] - 0.5, self.end[0] - 0.5), 
                                      1, 1, facecolor='red', alpha=0.8))
        
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        algorithm_name = self.algorithm if hasattr(self, 'algorithm') else 'Dijkstra'
        
        # 如果是RRT算法，绘制树结构
        if algorithm_name == 'RRT' and 'tree' in step:
            for node, parent in step['tree'].items():
                if parent is not None:
                    ax.plot([parent[1], node[1]], [parent[0], node[0]], 
                          'b-', linewidth=1, alpha=0.3)
        
        ax.set_title('{}算法搜索中... (步骤 {}/{})'.format(
            algorithm_name, step_num + 1, total_steps), fontsize=12, fontweight='bold')
        
        # 将matplotlib figure转换为PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
    
    def generate_gif(self):
        """根据实际步骤数生成GIF动图"""
        try:
            if self.maze is None:
                messagebox.showwarning("警告", "请先生成迷宫")
                return
            
            if not self.steps:
                # 如果还没有运行算法，先运行（但不显示动画）
                self.update_info("正在运行算法以生成步骤数据，请稍候...")
                self.root.update()
                try:
                    self.start_search_silent()
                except Exception as e:
                    messagebox.showerror("错误", "运行算法时出错: {}".format(str(e)))
                    import traceback
                    traceback.print_exc()
                    return
                
                if not self.steps:
                    messagebox.showerror("错误", "无法生成步骤数据，请先运行搜索算法")
                    return
            
            # 使用实际步骤数，不强制扩展
            original_total_steps = len(self.steps)
            total_steps = original_total_steps
            
            if total_steps == 0:
                messagebox.showerror("错误", "没有可用的步骤数据")
                return
            
            # 选择保存位置
            algorithm_name = self.algorithm if hasattr(self, 'algorithm') else 'Dijkstra'
            # 将算法名称中的非法字符替换为安全字符（用于文件名）
            # Windows文件名非法字符: < > : " / \ | ? *
            safe_algorithm_name = algorithm_name.replace('*', 'Star').replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            # 例如: "A*" -> "AStar", "RRT" -> "RRT"
            default_filename = "{}_动画_{}x{}_{}帧.gif".format(safe_algorithm_name, self.width, self.height, total_steps)
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".gif",
                filetypes=[("GIF files", "*.gif"), ("All files", "*.*")],
                initialfile=default_filename,
                title="保存GIF动图"
            )
            
            if not filepath:
                return
            
            # 计算是否需要采样以减少帧数（确保播放时间在10秒内且每帧时间合理）
            max_total_time_ms = 10000  # 10秒
            min_frame_duration_ms = 30  # 最小每帧30毫秒，保证可看性
            max_frames = max_total_time_ms // min_frame_duration_ms  # 最多约333帧
            
            steps_to_use = self.steps
            was_sampled = False
            if total_steps > max_frames:
                # 如果帧数太多，进行均匀采样
                was_sampled = True
                self.update_info("帧数过多（{}帧），正在进行采样以减少到{}帧以优化播放速度...".format(total_steps, max_frames))
                self.root.update()
                indices = np.linspace(0, total_steps - 1, max_frames, dtype=int)
                steps_to_use = [self.steps[i].copy() for i in indices]
                total_steps = len(steps_to_use)
            
            # 生成每一帧的图像
            if was_sampled:
                self.update_info("正在生成{}帧图像（已从{}帧采样）...".format(total_steps, original_total_steps))
            else:
                self.update_info("正在生成{}帧图像，请稍候...".format(total_steps))
            self.root.update()
            
            frames = []
            
            for i, step in enumerate(steps_to_use):
                try:
                    # 每10帧或每10%进度更新一次提示
                    if i % max(10, total_steps // 10) == 0 or i == 0:
                        progress = int((i + 1) / total_steps * 100)
                        self.update_info("正在生成第{}/{}帧 ({}%)...".format(i + 1, total_steps, progress))
                        self.root.update()
                    
                    img = self.draw_step_to_image(step, i, total_steps)
                    frames.append(img)
                except Exception as e:
                    error_msg = "生成第{}/{}帧时出错: {}".format(i + 1, total_steps, str(e))
                    messagebox.showerror("错误", error_msg)
                    import traceback
                    traceback.print_exc()
                    self.update_info("生成失败: {}".format(error_msg))
                    return
            
            # 添加最终结果帧（如果有路径）
            if self.path:
                self.update_info("正在添加最终路径帧...")
                self.root.update()
                # 创建最终结果图像
                fig, ax = plt.subplots(figsize=(10, 8), dpi=80)
                ax.imshow(self.maze, cmap='gray_r', interpolation='nearest')
                
                # 绘制已访问的节点
                for cell in self.visited:
                    if cell not in self.path:
                        ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, 
                                                 facecolor='steelblue', alpha=0.7))
                
                # 绘制路径
                for cell in self.path:
                    ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, 
                                             facecolor='yellow', alpha=0.8, edgecolor='orange', linewidth=2))
                
                # 绘制起点和终点
                if self.start:
                    ax.add_patch(plt.Rectangle((self.start[1] - 0.5, self.start[0] - 0.5), 
                                              1, 1, facecolor='green', alpha=0.8))
                if self.end:
                    ax.add_patch(plt.Rectangle((self.end[1] - 0.5, self.end[0] - 0.5), 
                                              1, 1, facecolor='red', alpha=0.8))
                
                ax.set_xlim(-0.5, self.width - 0.5)
                ax.set_ylim(self.height - 0.5, -0.5)
                ax.set_aspect('equal')
                ax.axis('off')
                
                path_length = len(self.path) - 1 if self.path else 0
                ax.set_title('搜索完成！最短路径长度: {}'.format(path_length), 
                           fontsize=12, fontweight='bold', color='green')
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1)
                buf.seek(0)
                final_img = Image.open(buf)
                plt.close(fig)
                
                # 添加最终帧多次，让它停留更久（根据总帧数决定停留帧数，但不超过总时长的20%）
                max_total_time_ms = 10000
                min_frame_duration_ms = 30
                # 最终结果帧应该占用约1-2秒
                max_final_frames = min(30, max(10, max_total_time_ms // (min_frame_duration_ms * 5)))
                final_frame_count = min(max_final_frames, max(10, total_steps // 10))
                for _ in range(final_frame_count):
                    frames.append(final_img)
            
            # 保存为GIF
            self.update_info("正在保存GIF文件，请稍候...")
            self.root.update()
            
            if frames:
                # 计算合适的每帧持续时间，确保总播放时间不超过10秒
                total_frames = len(frames)
                max_duration_ms = 10000  # 10秒 = 10000毫秒
                min_duration_ms = 20  # 最小20毫秒，避免太快看不清
                
                # 计算每帧持续时间
                calculated_duration = max_duration_ms / total_frames
                # 限制在合理范围内（20-100毫秒）
                frame_duration = max(min_duration_ms, min(100, int(calculated_duration)))
                
                # 计算实际总播放时间
                actual_total_time = total_frames * frame_duration / 1000.0  # 转换为秒
                
                try:
                    frames[0].save(
                        filepath,
                        save_all=True,
                        append_images=frames[1:],
                        duration=frame_duration,  # 动态计算的每帧持续时间
                        loop=0  # 无限循环
                    )
                    info_msg = "GIF动图已成功生成！\n"
                    if was_sampled:
                        info_msg += "原始帧数: {} (已采样到 {})\n".format(original_total_steps, total_steps)
                    info_msg += "总帧数: {}\n".format(total_frames)
                    info_msg += "每帧持续时间: {}毫秒\n".format(frame_duration)
                    info_msg += "总播放时间: {:.1f}秒\n".format(actual_total_time)
                    info_msg += "文件保存在: {}".format(filepath)
                    
                    messagebox.showinfo("完成", info_msg)
                    
                    info_text = "GIF动图生成完成！\n"
                    if was_sampled:
                        info_text += "原始步骤: {} (已采样到 {})\n".format(original_total_steps, total_steps)
                    info_text += "总帧数: {} (搜索步骤: {} + 最终结果: {})\n".format(
                        len(frames), total_steps, len(frames) - total_steps)
                    info_text += "每帧: {}毫秒\n".format(frame_duration)
                    info_text += "总时长: {:.1f}秒\n".format(actual_total_time)
                    info_text += "文件: {}".format(filepath)
                    self.update_info(info_text)
                except Exception as e:
                    error_msg = "保存GIF文件时出错: {}\n文件路径: {}".format(str(e), filepath)
                    messagebox.showerror("错误", error_msg)
                    import traceback
                    traceback.print_exc()
                    self.update_info("生成GIF失败: {}".format(str(e)))
            else:
                messagebox.showerror("错误", "无法生成GIF，没有可用的帧")
        except Exception as e:
            error_msg = "生成GIF时发生错误: {}".format(str(e))
            messagebox.showerror("错误", error_msg)
            import traceback
            traceback.print_exc()
            self.update_info("生成GIF失败: {}".format(str(e)))
    
    def start_search_silent(self):
        """静默运行搜索算法（不显示动画，只记录步骤）"""
        if self.maze is None:
            raise ValueError("迷宫未生成")
        
        if self.start is None or self.end is None:
            raise ValueError("起点或终点未设置")
        
        # 初始化
        self.distances = {self.start: 0}
        self.previous = {}
        self.visited = set()
        self.path = []
        self.steps = []
        self.current_step = 0
        
        # 运行算法
        try:
            if hasattr(self, 'algorithm_var'):
                self.algorithm = self.algorithm_var.get()
            elif hasattr(self, 'algorithm'):
                pass  # 使用已有的算法
            else:
                self.algorithm = 'Dijkstra'
        except Exception as e:
            self.algorithm = 'Dijkstra'
        
        # 确保算法变量已初始化
        if not hasattr(self, 'algorithm'):
            self.algorithm = 'Dijkstra'
        
        try:
            if self.algorithm == 'A*':
                self.run_astar_silent()
            elif self.algorithm == 'RRT':
                self.run_rrt_silent()
            else:
                self.run_dijkstra_silent()
        except Exception as e:
            raise RuntimeError("运行{}算法时出错: {}".format(self.algorithm, str(e)))
    
    def run_dijkstra_silent(self):
        """静默运行Dijkstra算法"""
        pq = [(0, self.start)]
        
        while pq:
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
            
            if current == self.end:
                break
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self.is_valid_cell(neighbor):
                    continue
                
                if neighbor in self.visited:
                    continue
                
                new_dist = current_dist + 1
                
                if neighbor not in self.distances or new_dist < self.distances[neighbor]:
                    self.distances[neighbor] = new_dist
                    self.previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        self.reconstruct_path()
    
    def run_astar_silent(self):
        """静默运行A*算法"""
        self.distances[self.start] = 0
        h_start = self.heuristic(self.start)
        f_start = 0 + self.astar_heuristic_weight * h_start
        pq = [(f_start, 0, self.start)]
        
        while pq:
            f_value, current_dist, current = heapq.heappop(pq)
            
            if current in self.visited:
                continue
            
            self.visited.add(current)
            
            self.steps.append({
                'current': current,
                'visited': self.visited.copy(),
                'distances': self.distances.copy()
            })
            
            if current == self.end:
                break
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self.is_valid_cell(neighbor):
                    continue
                
                if neighbor in self.visited:
                    continue
                
                new_dist = current_dist + 1
                
                if neighbor not in self.distances or new_dist < self.distances[neighbor]:
                    self.distances[neighbor] = new_dist
                    self.previous[neighbor] = current
                    h_value = self.heuristic(neighbor)
                    f_value = new_dist + self.astar_heuristic_weight * h_value
                    heapq.heappush(pq, (f_value, new_dist, neighbor))
        
        self.reconstruct_path()
    
    def run_rrt_silent(self):
        """静默运行RRT算法"""
        tree = {self.start: None}
        self.previous = {self.start: None}
        self.distances = {self.start: 0}
        self.visited = set([self.start])
        
        step_size = max(3, int(min(self.height, self.width) / self.rrt_step_size_factor))
        goal_threshold = step_size * 2
        max_iterations = self.rrt_max_iterations
        
        path_cells = []
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                if self.maze[i, j] == 0:
                    path_cells.append((i, j))
        
        if not path_cells:
            return
        
        iteration = 0
        steps_recorded = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            if iteration % 100 == 0:
                self.root.update()
            
            if random.random() < self.rrt_goal_bias:
                sample = self.end
            else:
                sample = random.choice(path_cells)
            
            min_dist = float('inf')
            nearest = self.start
            for node in tree.keys():
                dist_sq = (node[0] - sample[0])**2 + (node[1] - sample[1])**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    nearest = node
            
            dx = sample[0] - nearest[0]
            dy = sample[1] - nearest[1]
            dist = (dx**2 + dy**2)**0.5
            
            if dist < 0.1:
                continue
            
            candidates = [
                (nearest[0] + 1, nearest[1]),
                (nearest[0] - 1, nearest[1]),
                (nearest[0], nearest[1] + 1),
                (nearest[0], nearest[1] - 1),
            ]
            
            valid_candidates = [c for c in candidates if self.is_valid_cell(c)]
            
            if valid_candidates:
                new_node = min(valid_candidates, 
                             key=lambda c: ((c[0] - sample[0])**2 + (c[1] - sample[1])**2)**0.5)
            else:
                continue
            
            if not self.is_valid_cell(new_node):
                continue
            
            if abs(new_node[0] - nearest[0]) + abs(new_node[1] - nearest[1]) != 1:
                continue
            
            tree[new_node] = nearest
            self.previous[new_node] = nearest
            actual_dist = abs(new_node[0] - nearest[0]) + abs(new_node[1] - nearest[1])
            self.distances[new_node] = self.distances[nearest] + actual_dist
            self.visited.add(new_node)
            
            steps_recorded += 1
            if steps_recorded % 5 == 0:
                self.steps.append({
                    'current': new_node,
                    'visited': self.visited.copy(),
                    'distances': self.distances.copy(),
                    'tree': dict(tree)
                })
            
            manhattan_dist_to_goal = abs(new_node[0] - self.end[0]) + abs(new_node[1] - self.end[1])
            if manhattan_dist_to_goal <= goal_threshold:
                if self.is_path_clear_simple(new_node, self.end):
                    self.previous[self.end] = new_node
                    self.distances[self.end] = self.distances[new_node] + manhattan_dist_to_goal
                    self.visited.add(self.end)
                    self.steps.append({
                        'current': self.end,
                        'visited': self.visited.copy(),
                        'distances': self.distances.copy(),
                        'tree': dict(tree)
                    })
                    break
        
        if self.end not in self.previous and tree:
            closest_to_end = min(tree.keys(), 
                               key=lambda n: abs(n[0] - self.end[0]) + abs(n[1] - self.end[1]))
            manhattan_dist_to_end = abs(closest_to_end[0] - self.end[0]) + abs(closest_to_end[1] - self.end[1])
            if manhattan_dist_to_end <= goal_threshold * 3:
                if self.is_path_clear_simple(closest_to_end, self.end):
                    self.previous[self.end] = closest_to_end
                    self.distances[self.end] = self.distances[closest_to_end] + manhattan_dist_to_end
                    self.visited.add(self.end)
        
        self.reconstruct_path()

if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingMazeApp(root)
    root.mainloop()

