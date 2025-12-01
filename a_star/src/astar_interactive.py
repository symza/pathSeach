import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import defaultdict
import heapq
import matplotlib
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, FancyBboxPatch
import os
import sys

# 获取脚本所在目录和a_star文件夹路径
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

A_STAR_DIR = get_base_path()  # a_star文件夹（exe所在目录或脚本的上级目录）
IMAGES_DIR = os.path.join(A_STAR_DIR, 'images')  # images文件夹路径

# 确保images文件夹存在
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def euclidean_distance(pos1, pos2):
    """计算两点之间的欧几里得距离"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def manhattan_distance(pos1, pos2):
    """计算两点之间的曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class AStarInteractive:
    def __init__(self, grid_size=5, start_pos=(4, 0), goal_pos=(0, 4), obstacles=None, heuristic_type='manhattan'):
        """
        初始化A*算法网格可视化（交互式版本）
        
        参数:
            grid_size: 网格大小（默认5x5）
            start_pos: 起点坐标 (row, col)
            goal_pos: 终点坐标 (row, col)
            obstacles: 障碍物坐标列表 [(row, col), ...]
            heuristic_type: 启发式函数类型 ('euclidean' 或 'manhattan')
        """
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.heuristic_type = heuristic_type
        
        # 设置障碍物（如果没有提供，使用默认障碍物）
        if obstacles is None:
            # 默认障碍物：适配5x5网格
            obstacles = []
            # 水平障碍物
            obstacles.append((2, 1))
            obstacles.append((2, 2))
            # 垂直障碍物
            obstacles.append((1, 2))
            obstacles.append((3, 2))
        
        self.obstacles = set(obstacles)
        
        # 初始化g值和h值
        self.g_scores = {}
        self.h_scores = {}
        
        # 计算所有节点的h值
        for i in range(grid_size):
            for j in range(grid_size):
                pos = (i, j)
                if pos not in self.obstacles:
                    if heuristic_type == 'euclidean':
                        self.h_scores[pos] = euclidean_distance(pos, goal_pos)
                    else:
                        self.h_scores[pos] = manhattan_distance(pos, goal_pos)
                    self.g_scores[pos] = float('inf')
        
        # 起点g值为0
        self.g_scores[start_pos] = 0
        
        # 记录每一步的状态
        self.steps = []
        self.current_step = 0
        
        # 执行A*算法并记录每一步
        self.run_astar()
        
        # 创建图形（适合演示的尺寸，左侧演示图，右侧计算部分）
        self.fig, self.ax = plt.subplots(figsize=(28, 18), dpi=100)
        # 调整整个坐标轴在画布中的位置，让所有内容更贴近窗口边缘，
        # 但保持内部相对布局不变（左侧演示图、右侧说明和f框、图例、步骤信息的相对位置不变）
        self.fig.subplots_adjust(left=0.03, right=0.99, top=0.03, bottom=0.01)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.ax.set_facecolor('#f0f0f0')
        
        # 绘制初始状态
        self.draw_current_step()
    
    def is_valid_cell(self, pos):
        """检查单元格是否有效（在网格内且不是障碍物）"""
        i, j = pos
        return (0 <= i < self.grid_size and 
                0 <= j < self.grid_size and 
                pos not in self.obstacles)
    
    def get_neighbors(self, pos):
        """获取邻居节点（8方向）"""
        i, j = pos
        neighbors = []
        # 8方向：上下左右 + 四个对角线
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        
        for di, dj in directions:
            new_pos = (i + di, j + dj)
            if self.is_valid_cell(new_pos):
                # 对角线移动成本为√2，直线移动成本为1
                if abs(di) + abs(dj) == 2:
                    cost = np.sqrt(2)
                else:
                    cost = 1.0
                neighbors.append((new_pos, cost))
        
        return neighbors
    
    def run_astar(self):
        """执行A*算法并记录每一步的状态"""
        # 初始化
        visited = set()
        previous = {}  # 记录每个节点的最短路径来源
        # 优先级队列：(f_score, g_score, node)
        # f_score = g_score + h_score
        pq = [(self.h_scores[self.start_pos], 0, self.start_pos)]
        processed_nodes = set()
        old_previous = {}  # 记录更新前的路径，用于标记变灰的边
        
        # 记录初始状态
        self.steps.append({
            'g_scores': self.g_scores.copy(),
            'h_scores': self.h_scores.copy(),
            'current_node': None,
            'visited': visited.copy(),
            'processed': processed_nodes.copy(),
            'previous': previous.copy(),
            'checking_neighbor': None,
            'message': f'Initialization: Start ({self.start_pos[0]}, {self.start_pos[1]}) g(n)=0, h(n)={self.h_scores[self.start_pos]:.2f}, f(n)={self.h_scores[self.start_pos]:.2f}'
        })
        
        while pq:
            f_score, current_g, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            processed_nodes.add(current_node)
            
            # 记录选择当前节点
            self.steps.append({
                'g_scores': self.g_scores.copy(),
                'h_scores': self.h_scores.copy(),
                'current_node': current_node,
                'visited': visited.copy(),
                'processed': processed_nodes.copy(),
                'previous': previous.copy(),
                'checking_neighbor': None,
                'message': f'Select node ({current_node[0]}, {current_node[1]}) g(n)={current_g:.2f}, h(n)={self.h_scores[current_node]:.2f}, f(n)={f_score:.2f}'
            })
            
            # 如果到达目标节点，结束
            if current_node == self.goal_pos:
                self.steps.append({
                    'g_scores': self.g_scores.copy(),
                    'h_scores': self.h_scores.copy(),
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'processed': processed_nodes.copy(),
                    'previous': previous.copy(),
                    'checking_neighbor': None,
                    'message': f'Found goal ({self.goal_pos[0]}, {self.goal_pos[1]})! Path length: {current_g:.2f}'
                })
                break
            
            # 更新邻居节点
            neighbors = self.get_neighbors(current_node)
            for neighbor_pos, move_cost in neighbors:
                if neighbor_pos in visited:
                    continue
                
                tentative_g = current_g + move_cost
                
                # 记录检查邻居的过程（带上本次计算得到的 tentative g 值，方便右侧显示）
                self.steps.append({
                    'g_scores': self.g_scores.copy(),
                    'h_scores': self.h_scores.copy(),
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'processed': processed_nodes.copy(),
                    'previous': previous.copy(),
                    'checking_neighbor': neighbor_pos,
                    'checking_neighbor_g': tentative_g,
                    'message': f'Checking neighbor ({neighbor_pos[0]}, {neighbor_pos[1]}) from ({current_node[0]}, {current_node[1]})'
                })
                
                if tentative_g < self.g_scores[neighbor_pos]:
                    old_g = self.g_scores[neighbor_pos]
                    # 记录旧路径（如果存在）
                    old_prev = previous.get(neighbor_pos)
                    if old_prev is not None:
                        old_previous[neighbor_pos] = old_prev
                    
                    # 更新g值
                    self.g_scores[neighbor_pos] = tentative_g
                    previous[neighbor_pos] = current_node
                    
                    # 计算f值并加入优先队列
                    f_score_new = tentative_g + self.h_scores[neighbor_pos]
                    heapq.heappush(pq, (f_score_new, tentative_g, neighbor_pos))
                    
                    # 记录更新距离
                    step_data = {
                        'g_scores': self.g_scores.copy(),
                        'h_scores': self.h_scores.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'processed': processed_nodes.copy(),
                        'updated_node': neighbor_pos,
                        'old_g_score': old_g,
                        'new_g_score': tentative_g,
                        'previous': previous.copy(),
                        'old_previous': old_previous.copy() if old_prev is not None else {},
                        'checking_neighbor': None,
                        'message': f'Update node ({neighbor_pos[0]}, {neighbor_pos[1]}): g(n)={old_g if old_g == float("inf") else old_g:.2f}→{tentative_g:.2f}, h(n)={self.h_scores[neighbor_pos]:.2f}, f(n)={f_score_new:.2f}'
                    }
                    self.steps.append(step_data)
                else:
                    # 记录不更新
                    self.steps.append({
                        'g_scores': self.g_scores.copy(),
                        'h_scores': self.h_scores.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'processed': processed_nodes.copy(),
                        'previous': previous.copy(),
                        'checking_neighbor': None,
                        'message': (
                            f'Node ({neighbor_pos[0]}, {neighbor_pos[1]}) g not improved: '
                            f'{self.g_scores[neighbor_pos]:.2f} → {tentative_g:.2f}'
                        )
                    })
        
        # 记录完成状态
        if self.goal_pos in visited:
            # 重建路径
            path = []
            current = self.goal_pos
            while current is not None:
                path.append(current)
                current = previous.get(current)
            path.reverse()
            
            self.steps.append({
                'g_scores': self.g_scores.copy(),
                'h_scores': self.h_scores.copy(),
                'current_node': None,
                'visited': visited.copy(),
                'processed': processed_nodes.copy(),
                'previous': previous.copy(),
                'final_path': path,
                'checking_neighbor': None,
                'message': f'Algorithm complete! Total length: {self.g_scores[self.goal_pos]:.2f}\nShortest path: {" → ".join([f"({p[0]},{p[1]})" for p in path])}'
            })
        else:
            self.steps.append({
                'g_scores': self.g_scores.copy(),
                'h_scores': self.h_scores.copy(),
                'current_node': None,
                'visited': visited.copy(),
                'processed': processed_nodes.copy(),
                'previous': previous.copy(),
                'checking_neighbor': None,
                'message': 'Algorithm complete! No path found to goal'
            })
    
    def on_key_press(self, event):
        """处理键盘按键事件"""
        if event.key == 'right' or event.key == ' ' or event.key == 'n':
            # 前进到下一步
            self.next_step()
        elif event.key == 'left' or event.key == 'b':
            # 后退到上一步
            self.prev_step()
        elif event.key == 'home' or event.key == 'r':
            # 回到第一步
            self.first_step()
        elif event.key == 'end' or event.key == 'e':
            # 跳到最后一步
            self.last_step()
        elif event.key == 's':
            # 保存当前帧
            self.save_current_frame()
        elif event.key == 'escape':
            # 关闭窗口
            plt.close(self.fig)
    
    def on_mouse_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # 鼠标左键
            # 前进到下一步
            self.next_step()
        elif event.button == 3:  # 鼠标右键
            # 后退到上一步
            self.prev_step()
    
    def next_step(self):
        """前进到下一步"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.draw_current_step()
    
    def prev_step(self):
        """后退到上一步"""
        if self.current_step > 0:
            self.current_step -= 1
            self.draw_current_step()
    
    def first_step(self):
        """回到第一步"""
        self.current_step = 0
        self.draw_current_step()
    
    def last_step(self):
        """跳到最后一步"""
        self.current_step = len(self.steps) - 1
        self.draw_current_step()
    
    def save_current_frame(self):
        """保存当前帧为图片"""
        filename = os.path.join(IMAGES_DIR, f'astar_step_{self.current_step:03d}.png')
        self.fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"已保存当前帧: {filename}")
    
    def export_all_frames(self, output_dir=None):
        """导出所有帧为图片序列，供PPT使用"""
        if output_dir is None:
            output_dir = os.path.join(IMAGES_DIR, 'astar_frames')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"正在导出 {len(self.steps)} 帧到 {output_dir} 目录...")
        for i, step in enumerate(self.steps):
            self.current_step = i
            self.draw_current_step()
            filename = os.path.join(output_dir, f'step_{i:03d}.png')
            self.fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
            if (i + 1) % 10 == 0:
                print(f"  已导出 {i + 1}/{len(self.steps)} 帧...")
        
        print(f"✓ 所有帧已导出到 {output_dir} 目录")
        print(f"  可以在PPT中按顺序插入这些图片，实现手动控制播放")
    
    def draw_current_step(self):
        """绘制当前步骤的网格"""
        if self.current_step >= len(self.steps):
            return
        
        step_data = self.steps[self.current_step]
        self.ax.clear()
        self.ax.set_facecolor('#f0f0f0')
        
        # 设置坐标范围：上方图例，左侧演示图，右侧计算部分，下方步骤信息
        # 增大网格显示区域，使演示图更大
        grid_scale = 2.5  # 网格放大倍数（从1.5增加到2.5）
        left_margin = 1.0
        right_margin = 12  # 右侧留出空间显示计算部分（增大以容纳放大后的文字）
        top_margin = 4.5  # 上方留出空间显示图例（增大以容纳放大后的文字和避免被边界遮住）
        bottom_margin = 4.0  # 下方留出空间显示步骤信息（增大以容纳放大后的文字）
        separation = 0.5  # 左右两部分之间的间距（再减小，让中间内容整体更靠左）
        
        # 仅对左侧演示图做水平偏移（负数表示向左），右侧说明区域保持不变
        grid_offset = -4.0
    
        
        # 使用更大的坐标范围来放大网格显示
        # 左边界与 grid_offset 对齐，这样左侧不再额外留“空气”（空白），
        # 同时仍能完整显示最左边一列网格。
        scaled_grid_size = self.grid_size * grid_scale
        self.ax.set_xlim(grid_offset, scaled_grid_size + separation + right_margin)
        self.ax.set_ylim(-bottom_margin, scaled_grid_size + top_margin)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        g_scores = step_data['g_scores']
        h_scores = step_data['h_scores']
        current_node = step_data.get('current_node')
        processed = step_data.get('processed', set())
        checking_neighbor = step_data.get('checking_neighbor')
        updated_node = step_data.get('updated_node')
        previous = step_data.get('previous', {})
        final_path = step_data.get('final_path', [])
        visited = step_data.get('visited', set())
        # 对于“正在检查”的邻居，记录本步计算得到的 tentative g 值（如果有）
        next_g_override = step_data.get('checking_neighbor_g')
        
        # 网格放大倍数，使演示图更大
        grid_scale = 2.5
        
        # 绘制网格线（放大后的网格，整体向左偏移 grid_offset）
        for i in range(self.grid_size + 1):
            x = i * grid_scale + grid_offset
            y = i * grid_scale
            self.ax.plot([x, x], [0, self.grid_size * grid_scale], 'k-', linewidth=0.5, alpha=0.3)
            self.ax.plot([grid_offset, grid_offset + self.grid_size * grid_scale], [y, y], 'k-', linewidth=0.5, alpha=0.3)
        
        # 绘制每个单元格
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = (i, j)
                
                # 确定单元格颜色
                if pos == self.start_pos:
                    cell_color = 'green'  # 起点：绿色
                elif pos == self.goal_pos:
                    cell_color = 'green'  # 终点：绿色
                elif pos in self.obstacles:
                    cell_color = 'blue'  # 障碍物：蓝色
                elif pos in final_path:
                    cell_color = 'yellow'  # 最终路径：黄色
                elif pos == current_node:
                    cell_color = 'yellow'  # 当前节点：黄色
                elif pos == checking_neighbor:
                    cell_color = 'lightblue'  # 正在检查的邻居：浅蓝色
                elif pos == updated_node:
                    cell_color = 'lightblue'  # 刚更新的节点：浅蓝色
                elif pos in processed:
                    cell_color = 'lightgreen'  # 已处理的节点：浅绿色
                elif pos in visited:
                    cell_color = 'yellow'  # 已访问的节点：黄色
                else:
                    cell_color = 'white'  # 未访问的节点：白色
                
                # 绘制单元格（放大后的位置，并整体向左偏移 grid_offset）
                rect = Rectangle((j * grid_scale + grid_offset, (self.grid_size - 1 - i) * grid_scale), 
                               grid_scale, grid_scale,
                               facecolor=cell_color, edgecolor='black', 
                               linewidth=1, alpha=0.8)
                self.ax.add_patch(rect)
                
                # 如果不是障碍物，只显示f(n)值
                if pos not in self.obstacles:
                    g_score = g_scores.get(pos, float('inf'))
                    h_score = h_scores.get(pos, float('inf'))
                    
                    # 格式化显示f值
                    if g_score == float('inf'):
                        f_text = '∞'
                    else:
                        f_score = g_score + h_score
                        f_text = f'{f_score:.1f}'
                    
                    # 显示f(n)在中心（放大后的位置，并整体向左偏移 grid_offset）
                    center_x = (j + 0.5) * grid_scale + grid_offset
                    center_y = (self.grid_size - 1 - i + 0.5) * grid_scale
                    
                    self.ax.text(center_x, center_y, f'f={f_text}',
                               fontsize=20, ha='center', va='center', fontweight='bold',
                               color='red' if pos == current_node or pos == updated_node or pos == checking_neighbor else 'black')
                
                # 添加特殊标签（放大后的位置，并整体向左偏移 grid_offset）
                if pos == self.start_pos:
                    self.ax.text((j + 0.5) * grid_scale + grid_offset, (self.grid_size - 1 - i + 0.5) * grid_scale, 
                               f'({i},{j})\nStart', fontsize=20, ha='center', va='center',
                               fontweight='bold', color='white',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', 
                                       edgecolor='darkgreen', alpha=0.8, linewidth=2))
                elif pos == self.goal_pos:
                    self.ax.text((j + 0.5) * grid_scale + grid_offset, (self.grid_size - 1 - i + 0.5) * grid_scale, 
                               f'({i},{j})\nGoal', fontsize=20, ha='center', va='center',
                               fontweight='bold', color='white',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', 
                                       edgecolor='darkgreen', alpha=0.8, linewidth=2))
                elif pos == checking_neighbor:
                    self.ax.text((j + 0.5) * grid_scale + grid_offset, (self.grid_size - 1 - i + 0.5) * grid_scale, 
                               f'({i},{j})\nNeighbor', fontsize=18, ha='center', va='center',
                               fontweight='bold', color='black',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                                       edgecolor='blue', alpha=0.8, linewidth=2))
        
        # 网格放大倍数（与上面保持一致）
        grid_scale = 2.5
        scaled_grid_size = self.grid_size * grid_scale
        
        # 添加图例（在网格上方，居中，调整位置避免被边界遮住）
        legend_text = "Legend: Green=Start/Goal | Blue=Obstacle | Yellow=Explored/Path | Light Blue=Current/Neighbor | Light Green=Processed"
        # 将图例位置向下移动，确保不被边界遮住
        legend_y = scaled_grid_size + top_margin - 2.0
        self.ax.text((scaled_grid_size + separation + right_margin) / 2, legend_y, legend_text, 
                    fontsize=16, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='gray', alpha=0.9, linewidth=2),
                    fontweight='bold')
        
        # 添加步骤信息（在网格下方，居中）
        message = step_data.get('message', '')
        # 将步骤信息位置进一步向下移动，确保不与右侧计算面板的绿色框重叠
        # 保证仍在坐标范围内（大于 -bottom_margin）
        step_y = -bottom_margin + 0.3
        self.ax.text((scaled_grid_size + separation + right_margin) / 2, step_y, message,
                    fontsize=24, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow',
                            edgecolor='black', alpha=0.95, linewidth=3),
                    fontweight='bold')
        
        # 在右侧显示f值（只显示三个节点：当前、下一个、上一个）
        # 获取上一个节点（从previous字典中找到当前节点的前一个节点）
        previous_node = None
        if current_node is not None and previous:
            previous_node = previous.get(current_node)
        
        self.draw_f_values_panel(g_scores, h_scores, current_node, checking_neighbor, previous_node, next_g_override)
        
        self.fig.canvas.draw()
    
    def draw_f_values_panel(self, g_scores, h_scores, current_node, next_node, previous_node, next_g_override=None):
        """在右侧绘制f值面板，只显示三个节点：当前（黄）、下一个（蓝）、上一个（绿）
        
        参数:
            g_scores: 当前步骤的 g 值字典
            h_scores: 当前步骤的 h 值字典
            current_node: 当前节点（黄色）
            next_node: 正在检查的邻居节点（蓝色）
            previous_node: 当前节点的前驱节点（绿色）
            next_g_override: 对于 next_node，如果提供，则使用该 tentative g 值，
                             而不是 g_scores[next_node]（避免显示为 ∞ + h 而看不懂）
        """
        # 网格放大倍数（与draw_current_step保持一致）
        grid_scale = 2.5
        scaled_grid_size = self.grid_size * grid_scale
        
        # 右侧面板位置（在分隔线右侧，进一步减小间距，让中间内容整体更靠左）
        separation = 0.5
        panel_x = scaled_grid_size + separation + 0.5
        
        # 计算面板总高度（说明 + 标题 + 三个节点）
        # 进一步增大间距以避免重叠，只调整位置，不改变大小
        explanation_height = 3.5  # 说明文字高度（4行文字，字体20，需要更多空间）
        title_height = 2.0  # 标题高度（字体28，需要更多空间）
        node_height = 2.0  # 每个节点的高度（2行文字，字体22，需要更多空间）
        spacing_between = 0.8  # 元素之间的额外间距（进一步增大以避免重叠）
        
        # 不再按“居中 + 偏移”，而是从一个靠上的固定位置往下排，
        # 这样可以保证右侧面板整体远离底部步骤信息，不会再重叠。
        # 右侧面板顶部放在放大后网格的上方一点。
        panel_y_start = scaled_grid_size + 1.0
        
        # 添加说明文字（f、g、h的含义和h使用的距离）
        heuristic_name = 'Euclidean' if self.heuristic_type == 'euclidean' else 'Manhattan'
        explanation_text = f"f(n): total cost from start to goal through n\n"
        explanation_text += f"g(n): actual cost from start to n\n"
        explanation_text += f"h(n): estimated cost from n to goal\n"
        explanation_text += f"h uses: {heuristic_name} distance"
        
        current_y = panel_y_start
        self.ax.text(panel_x, current_y, explanation_text,
                    fontsize=20, ha='left', va='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                            edgecolor='orange', alpha=0.9, linewidth=2))
        
        # 标题（向下移动，增加间距）
        current_y = current_y - explanation_height - spacing_between
        self.ax.text(panel_x, current_y, 'f(n) = g(n) + h(n)', 
                    fontsize=28, ha='left', va='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', 
                            edgecolor='blue', alpha=0.9, linewidth=2))
        
        # 三个节点的起始位置（进一步增加间距）
        y_pos = current_y - title_height - spacing_between
        line_height = node_height + spacing_between
        
        # 显示三个节点的f值
        nodes_to_show = [
            ('Current', current_node, 'yellow', 'orange', 'red'),
            ('Next', next_node, 'lightblue', 'blue', 'blue'),
            ('Previous', previous_node, 'lightgreen', 'green', 'darkgreen'),
        ]
        
        for label, pos, bg_color, edge_color, text_color in nodes_to_show:
            if pos is not None:
                i, j = pos
                # 默认使用当前步骤记录的 g 值
                g_score = g_scores.get(pos, float('inf'))
                # 对于“正在检查”的邻居，优先显示本步计算得到的 tentative g，
                # 避免出现 g=∞ + h 的形式，影响理解。
                if label == 'Next' and next_g_override is not None and pos == next_node:
                    g_score = next_g_override
                h_score = h_scores.get(pos, float('inf'))
                
                if g_score == float('inf'):
                    f_text = '∞'
                    g_text = '∞'
                else:
                    f_score = g_score + h_score
                    f_text = f'{f_score:.2f}'
                    g_text = f'{g_score:.2f}'
                
                # 显示格式：位置 (i,j): f(n) = g(n) + h(n) = f值
                display_text = f"{label} ({i},{j}):\nf = {g_text} + {h_score:.2f} = {f_text}"
                
                self.ax.text(panel_x, y_pos, display_text,
                            fontsize=22, ha='left', va='top', fontweight='bold',
                            color=text_color,
                            bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_color, 
                                    edgecolor=edge_color, alpha=0.8, linewidth=2))
                
                # 移动到下一个节点位置，确保有足够间距
                y_pos -= line_height
    
    def show(self):
        """显示交互式窗口"""
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    try:
        # 创建交互式可视化对象
        print("=" * 60)
        print("A* Algorithm Interactive Visualization")
        print("=" * 60)
        print("\nInitializing grid structure...")
        print("Grid size: 5x5")
        print("Start: (4, 0), Goal: (0, 4)")
        print("Heuristic: Manhattan distance")
        
        viz = AStarInteractive(grid_size=5, start_pos=(4, 0), goal_pos=(0, 4), heuristic_type='manhattan')
        
        print(f"\nTotal steps: {len(viz.steps)}")
        print("\nControls:")
        print("  → / Space / Left Click: Next step")
        print("  ← / B / Right Click: Previous step")
        print("  Home / R: First step")
        print("  End / E: Last step")
        print("  S: Save current frame")
        print("  Esc: Exit")
        print("\n" + "=" * 60)
        
        # 可选：导出所有帧为图片序列（取消注释以使用）
        # viz.export_all_frames('astar_frames')
        
        # 显示交互式窗口
        viz.show()
    except Exception as e:
        import traceback
        error_msg = f"程序运行出错:\n{str(e)}\n\n详细错误信息:\n{traceback.format_exc()}"
        print(error_msg)
        # 如果是打包后的exe，尝试显示错误对话框
        if getattr(sys, 'frozen', False):
            try:
                import tkinter.messagebox as messagebox
                messagebox.showerror("错误", error_msg)
            except:
                pass
        input("\n按回车键退出...")
        sys.exit(1)

