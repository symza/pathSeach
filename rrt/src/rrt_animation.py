import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import math
import os
import sys
from datetime import datetime
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection

# 获取脚本所在目录和rrt文件夹路径
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

RRT_DIR = get_base_path()  # rrt文件夹（exe所在目录或脚本的上级目录）
IMAGES_DIR = os.path.join(RRT_DIR, 'images')  # images文件夹路径

# 确保images文件夹存在
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RRTAnimation:
    def __init__(self, start=(50, 50), goal=(450, 450), 
                 obstacles=None, step_size=20, goal_radius=30, 
                 max_iter=1000, world_size=(500, 500), random_seed=None):
        """
        RRT算法动画可视化
        
        参数:
            start: 起点坐标 (x, y)
            goal: 终点坐标 (x, y)
            obstacles: 障碍物列表 [(x, y, radius), ...]
            step_size: 每次扩展的步长
            goal_radius: 目标点半径
            max_iter: 最大迭代次数
            world_size: 世界大小 (width, height)
            random_seed: 随机种子（用于重现结果）
        """
        # 设置随机种子（如果提供）
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_iter = max_iter
        self.world_width, self.world_height = world_size
        self.random_seed = random_seed
        
        # 设置障碍物
        if obstacles is None:
            obstacles = [
                (200, 200, 50),
                (300, 300, 40),
                (100, 400, 60),
                (400, 100, 45)
            ]
        self.obstacles = obstacles
        
        # 初始化树
        self.tree = []
        self.tree.append({'point': self.start, 'parent': -1})
        self.path = []
        self.found_path = False
        
        # 记录每一步的状态
        self.steps = []
        
        # 执行RRT算法并记录每一步
        self.run_rrt()
        
        # 创建图形（适合演示的尺寸）
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.ax.set_xlim(0, self.world_width)
        self.ax.set_ylim(0, self.world_height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f0f0f0')
        self.ax.set_title('RRT Algorithm Animation', fontsize=28, fontweight='bold')
    
    def distance(self, p1, p2):
        """计算两点之间的欧几里得距离"""
        return np.linalg.norm(p1 - p2)
    
    def nearest_node(self, point):
        """找到树中离给定点最近的节点"""
        min_dist = float('inf')
        min_index = -1
        
        for i, node in enumerate(self.tree):
            dist = self.distance(node['point'], point)
            if dist < min_dist:
                min_dist = dist
                min_index = i
                
        return min_index, min_dist
    
    def steer(self, from_point, to_point):
        """从from_point向to_point方向移动step_size距离"""
        direction = to_point - from_point
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return to_point
        else:
            return from_point + (direction / dist) * self.step_size
    
    def collision_free(self, point1, point2):
        """检查两点之间的路径是否与障碍物碰撞"""
        for (x, y, r) in self.obstacles:
            obstacle = np.array([x, y])
            
            # 检查线段到圆心的最短距离
            v = point2 - point1
            w = obstacle - point1
            
            c1 = np.dot(w, v)
            if c1 <= 0:
                dist = self.distance(obstacle, point1)
            else:
                c2 = np.dot(v, v)
                if c2 <= c1:
                    dist = self.distance(obstacle, point2)
                else:
                    b = c1 / c2
                    pb = point1 + b * v
                    dist = self.distance(obstacle, pb)
            
            if dist < r:
                return False
                
        return True
    
    def run_rrt(self):
        """执行RRT算法并记录每一步的状态"""
        for i in range(self.max_iter):
            # 随机采样点 (90%随机点，10%目标点)
            if random.random() < 0.1:
                rand_point = self.goal
            else:
                rand_point = np.array([
                    random.uniform(0, self.world_width),
                    random.uniform(0, self.world_height)
                ])
            
            # 找到最近的节点
            nearest_idx, _ = self.nearest_node(rand_point)
            nearest_node = self.tree[nearest_idx]
            
            # 向随机点方向扩展
            new_point = self.steer(nearest_node['point'], rand_point)
            
            # 检查路径是否无碰撞
            if self.collision_free(nearest_node['point'], new_point):
                # 添加新节点到树
                new_node = {
                    'point': new_point,
                    'parent': nearest_idx
                }
                self.tree.append(new_node)
                
                # 记录当前步骤状态
                step_data = {
                    'tree': self.tree.copy(),
                    'rand_point': rand_point,
                    'nearest_node': nearest_node,
                    'new_point': new_point,
                    'iteration': i,
                    'message': f'Iteration {i+1}: Add new node ({int(new_point[0])}, {int(new_point[1])})'
                }
                self.steps.append(step_data)
                
                # 检查是否到达目标
                if self.distance(new_point, self.goal) < self.goal_radius:
                    self.found_path = True
                    # 重建路径
                    self.path = []
                    current_idx = len(self.tree) - 1
                    while current_idx != -1:
                        self.path.append(self.tree[current_idx]['point'])
                        current_idx = self.tree[current_idx]['parent']
                    self.path.reverse()
                    
                    step_data = {
                        'tree': self.tree.copy(),
                        'path': self.path.copy(),
                        'iteration': i,
                        'message': f'Path found! Iterations: {i+1}, Path length: {len(self.path)}'
                    }
                    self.steps.append(step_data)
                    break
        
        if not self.found_path:
            step_data = {
                'tree': self.tree.copy(),
                'iteration': self.max_iter,
                'message': f'Path not found (reached max iterations {self.max_iter})'
            }
            self.steps.append(step_data)
    
    def draw_frame(self, step_data):
        """绘制单帧"""
        self.ax.clear()
        
        # 设置坐标范围和标题
        self.ax.set_xlim(0, self.world_width)
        self.ax.set_ylim(0, self.world_height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f0f0f0')
        self.ax.set_title('RRT Algorithm Animation', fontsize=28, fontweight='bold')
        
        # 绘制障碍物
        for (x, y, r) in self.obstacles:
            circle = Circle((x, y), r, color='blue', alpha=0.6)
            self.ax.add_patch(circle)
        
        # 绘制起点和终点
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=15, label='Start')
        self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=15, label='Goal')
        
        # 绘制目标区域
        goal_circle = Circle((self.goal[0], self.goal[1]), self.goal_radius, 
                            color='red', alpha=0.2, linestyle='--')
        self.ax.add_patch(goal_circle)
        
        # 绘制树
        lines = []
        for i, node in enumerate(step_data['tree']):
            if node['parent'] != -1:
                parent_node = step_data['tree'][node['parent']]
                lines.append([parent_node['point'], node['point']])
        
        if lines:
            lc = LineCollection(lines, colors='gray', linewidths=1, alpha=0.7)
            self.ax.add_collection(lc)
        
        # 绘制所有节点
        for node in step_data['tree']:
            self.ax.plot(node['point'][0], node['point'][1], 'o', 
                        color='#1f77b4', markersize=4, alpha=0.7)
        
        # 绘制当前步骤的特殊点
        if 'rand_point' in step_data:
            self.ax.plot(step_data['rand_point'][0], step_data['rand_point'][1], 
                        'b*', markersize=12, label='Random Point')
        
        if 'nearest_node' in step_data:
            nearest = step_data['nearest_node']['point']
            self.ax.plot(nearest[0], nearest[1], 'yo', markersize=10, label='Nearest Node')
        
        if 'new_point' in step_data:
            new_point = step_data['new_point']
            self.ax.plot(new_point[0], new_point[1], 'mo', markersize=8, label='New Node')
            
            # 绘制新节点到最近节点的连线
            if 'nearest_node' in step_data:
                nearest = step_data['nearest_node']['point']
                self.ax.plot([nearest[0], new_point[0]], [nearest[1], new_point[1]], 
                            'm-', linewidth=2, alpha=0.8)
        
        # 绘制路径（如果存在）
        if 'path' in step_data and step_data['path']:
            path = step_data['path']
            self.ax.plot([p[0] for p in path], [p[1] for p in path], 
                        'r-', linewidth=2, label='Path')
            self.ax.plot([p[0] for p in path], [p[1] for p in path], 
                        'ro', markersize=4)
        
        # 添加图例（放在右下角）
        self.ax.legend(loc='lower right', fontsize=18, framealpha=0.9)
        
        # 添加步骤信息
        message = step_data.get('message', '')
        self.ax.text(0.5, 0.02, message, transform=self.ax.transAxes,
                    fontsize=24, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                            edgecolor='black', alpha=0.8))
        
        # 添加迭代信息
        iter_info = f"Iteration: {step_data['iteration']+1}/{self.max_iter}"
        self.ax.text(0.02, 0.98, iter_info, transform=self.ax.transAxes,
                    fontsize=20, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.7))
        
        # 添加节点数量信息
        node_count = f"Nodes: {len(step_data['tree'])}"
        self.ax.text(0.98, 0.98, node_count, transform=self.ax.transAxes,
                    fontsize=20, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.7))
    
    def animate(self, frame):
        """动画函数"""
        if frame < len(self.steps):
            self.draw_frame(self.steps[frame])
        return []
    
    def generate_filename(self):
        """生成文件名，包含关键信息以区分不同的运行结果
        
        命名格式: RRT_动画_{world_size}x{world_size}_{frames}帧_{iterations}迭代_{nodes}节点_{path_length}路径.gif
        如果未找到路径，则: RRT_动画_{world_size}x{world_size}_{frames}帧_{iterations}迭代_{nodes}节点_未找到路径.gif
        """
        world_size_str = f"{self.world_width}x{self.world_height}"
        frames = len(self.steps)
        final_iteration = self.steps[-1]['iteration'] + 1 if self.steps else 0
        final_nodes = len(self.steps[-1]['tree']) if self.steps else 0
        
        if self.found_path and self.path:
            path_length = len(self.path)
            filename = f"RRT_动画_{world_size_str}_{frames}帧_{final_iteration}迭代_{final_nodes}节点_{path_length}路径.gif"
        else:
            filename = f"RRT_动画_{world_size_str}_{frames}帧_{final_iteration}迭代_{final_nodes}节点_未找到路径.gif"
        
        return filename
    
    def create_animation(self, output_file=None, interval=None, dpi=100):
        """创建动画并保存为GIF
        
        参数:
            output_file: 输出文件名，如果为None则自动生成
            interval: 每帧间隔时间（毫秒），如果为None则自动计算以限制在10秒内
            dpi: 图像分辨率（默认100）
        """
        # 不设置全局字体大小，让各个text元素使用自己指定的fontsize
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        
        # 如果没有指定输出文件，使用自动生成的文件名
        if output_file is None:
            output_file = os.path.join(IMAGES_DIR, self.generate_filename())
        
        total_frames = len(self.steps)
        print(f"Generating animation with {total_frames} frames...")
        print(f"This may take some time, please wait...")
        print(f"Note: Generating high-resolution animation requires more time, please be patient...")
        
        # 自动计算帧间隔，确保总播放时间不超过10秒
        max_total_time_ms = 10000  # 10秒 = 10000毫秒
        min_frame_duration_ms = 20  # 最小每帧20毫秒，避免太快看不清
        
        if interval is None or total_frames * interval > max_total_time_ms:
            # 计算合适的帧间隔
            calculated_interval = max_total_time_ms / total_frames
            # 限制在合理范围内（20-100毫秒）
            interval = max(min_frame_duration_ms, min(100, int(calculated_interval)))
            print(f"Auto-calculated frame interval: {interval}ms to fit within 10 seconds")
        
        actual_total_time = total_frames * interval / 1000.0  # 转换为秒
        print(f"Total animation duration: {actual_total_time:.1f} seconds")
        
        anim = animation.FuncAnimation(self.fig, self.animate, 
                                      frames=total_frames,
                                      interval=interval, 
                                      repeat=True, 
                                      blit=False)
        
        fps = 1000 / interval if interval > 0 else 10
        anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
        
        print(f"\n✓ Animation successfully saved as: {output_file}")
        print(f"  - Total frames: {len(self.steps)}")
        print(f"  - Resolution: {dpi} DPI")
        print(f"  - Frame interval: {interval}ms")
        print(f"  - Total duration: {actual_total_time:.1f} seconds")
        print(f"  - Final nodes: {len(self.steps[-1]['tree']) if self.steps else 0}")
        print(f"  - Path found: {self.found_path}")
        if self.found_path and self.path:
            print(f"  - Path length: {len(self.path)} nodes")
        
        return anim

if __name__ == '__main__':
    try:
        print("=" * 60)
        print("RRT Algorithm Animation Generator")
        print("=" * 60)
        
        # 创建可视化对象
        print("\nInitializing RRT parameters...")
        print("Start: (50, 50), Goal: (450, 450)")
        print("Step size: 20, Goal radius: 30")
        print("Max iterations: 1000")
        print("World size: 500x500")
        
        # 可以设置random_seed来重现结果，或者不设置让每次运行都不同
        viz = RRTAnimation(
            start=(50, 50), 
            goal=(450, 450),
            step_size=20,
            goal_radius=30,
            max_iter=1000,
            world_size=(500, 500),
            random_seed=None  # 设置为None让每次运行都不同，或设置固定值重现结果
        )
        
        # 生成动画
        print("\nStarting animation generation...")
        anim = viz.create_animation(interval=None, dpi=100)
        
        print("\n" + "=" * 60)
        print("Complete!")
        print("=" * 60)
    except Exception as e:
        import traceback
        error_msg = f"Program error:\n{str(e)}\n\nDetailed error information:\n{traceback.format_exc()}"
        print(error_msg)
        # 如果是打包后的exe，尝试显示错误对话框
        if getattr(sys, 'frozen', False):
            try:
                import tkinter.messagebox as messagebox
                messagebox.showerror("Error", error_msg)
            except:
                pass
        input("\nPress Enter to exit...")
        sys.exit(1)

