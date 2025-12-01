import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os
import sys
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

class RRTInteractive:
    def __init__(self, start=(50, 50), goal=(450, 450), 
                 obstacles=None, step_size=20, goal_radius=30, 
                 max_iter=1000, world_size=(500, 500)):
        """
        RRT算法交互式可视化
        
        参数:
            start: 起点坐标 (x, y)
            goal: 终点坐标 (x, y)
            obstacles: 障碍物列表 [(x, y, radius), ...]
            step_size: 每次扩展的步长
            goal_radius: 目标点半径
            max_iter: 最大迭代次数
            world_size: 世界大小 (width, height)
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_iter = max_iter
        self.world_width, self.world_height = world_size
        
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
        self.current_step = 0
        
        # 执行RRT算法并记录每一步
        self.run_rrt()
        
        # 创建图形（适合演示的尺寸）
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.ax.set_xlim(0, self.world_width)
        self.ax.set_ylim(0, self.world_height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f0f0f0')
        self.ax.set_title('RRT Algorithm Interactive Visualization', fontsize=20, fontweight='bold')
        
        # 绘制初始状态
        self.draw_current_step()
    
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
        filename = os.path.join(IMAGES_DIR, f'rrt_step_{self.current_step:03d}.png')
        self.fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved current frame: {filename}")
    
    def export_all_frames(self, output_dir=None):
        """导出所有帧为图片序列，供PPT使用"""
        if output_dir is None:
            output_dir = os.path.join(IMAGES_DIR, 'rrt_frames')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Exporting {len(self.steps)} frames to {output_dir}...")
        for i, step in enumerate(self.steps):
            self.current_step = i
            self.draw_current_step()
            filename = os.path.join(output_dir, f'step_{i:03d}.png')
            self.fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
            if (i + 1) % 10 == 0:
                print(f"  Exported {i + 1}/{len(self.steps)} frames...")
        
        print(f"✓ All frames exported to {output_dir}")
        print(f"  You can insert these images in PPT in order for manual playback")
    
    def draw_current_step(self):
        """绘制当前步骤的状态"""
        if self.current_step >= len(self.steps):
            return
        
        step_data = self.steps[self.current_step]
        self.ax.clear()
        
        # 设置坐标范围和标题
        self.ax.set_xlim(0, self.world_width)
        self.ax.set_ylim(0, self.world_height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f0f0f0')
        self.ax.set_title('RRT Algorithm Interactive Visualization', fontsize=20, fontweight='bold')
        
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
        
        # 添加图例（放在右下角，避免遮挡迭代信息、节点数量和起点终点）
        self.ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        
        # 添加步骤信息
        message = step_data.get('message', '')
        self.ax.text(0.5, 0.02, message, transform=self.ax.transAxes,
                    fontsize=16, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                            edgecolor='black', alpha=0.8))
        
        # 添加迭代信息
        iter_info = f"Iteration: {step_data['iteration']+1}/{self.max_iter}"
        self.ax.text(0.02, 0.98, iter_info, transform=self.ax.transAxes,
                    fontsize=14, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.7))
        
        # 添加节点数量信息
        node_count = f"Nodes: {len(step_data['tree'])}"
        self.ax.text(0.98, 0.98, node_count, transform=self.ax.transAxes,
                    fontsize=14, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.7))
        
        self.fig.canvas.draw()
    
    def show(self):
        """显示交互式窗口"""
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    try:
        # 创建交互式可视化对象
        print("=" * 60)
        print("RRT Algorithm Interactive Visualization")
        print("=" * 60)
        print("\nParameters:")
        print(f"Start: (50, 50), Goal: (450, 450)")
        print(f"Step size: 20, Goal radius: 30")
        print(f"Max iterations: 1000")
        print(f"World size: 500x500")
        
        viz = RRTInteractive()
        
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
        # viz.export_all_frames('rrt_frames')
        
        # 显示交互式窗口
        viz.show()
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