import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from collections import defaultdict
import heapq
import matplotlib
from matplotlib.colors import to_rgb
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def interpolate_color(color1, color2, alpha):
    """在两个颜色之间插值
    alpha: 0.0 表示 color1, 1.0 表示 color2
    """
    rgb1 = np.array(to_rgb(color1))
    rgb2 = np.array(to_rgb(color2))
    rgb = rgb1 * (1 - alpha) + rgb2 * alpha
    return tuple(rgb)

class DijkstraInteractive:
    def __init__(self):
        # 创建有向图
        self.G = nx.DiGraph()
        
        # 添加边（源节点，目标节点，权重）
        edges = [
            (0, 2, 2),
            (0, 1, 5),
            (2, 5, 8),
            (2, 3, 6),
            (1, 3, 1),
            (1, 4, 6),
            (3, 5, 2),
            (3, 4, 1),
            (5, 6, 3),
            (4, 6, 1)
        ]
        
        self.G.add_weighted_edges_from(edges)
        
        # 设置节点位置（保持原样式，使用原始位置，不向右平移）
        # 原位置：0:(0,0), 1:(2,1), 2:(2,-1), 3:(4,0), 4:(4,1), 5:(6,-1), 6:(6,0)
        # 使用原始位置（再向左移动1个单位）
        self.pos = {
            0: (0, 0),
            1: (2, 1),
            2: (2, -1),
            3: (4, 0),
            4: (4, 1),
            5: (6, -1),
            6: (6, 0)
        }
        
        # 初始化距离字典
        self.distances = {node: float('inf') for node in self.G.nodes()}
        self.distances[0] = 0  # 源点距离为0
        
        # 记录每一步的状态
        self.steps = []
        self.current_step = 0
        
        # 执行Dijkstra算法并记录每一步
        self.run_dijkstra()
        
        # 创建图形（适合演示的尺寸，左侧留出表格空间）
        # 原图形是(16, 10)，现在需要更宽以容纳左侧表格
        self.fig, self.ax = plt.subplots(figsize=(22, 10), dpi=100)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.ax.set_facecolor('#f0f0f0')
        
        # 绘制初始状态
        self.draw_current_step()
        
    def run_dijkstra(self):
        """执行Dijkstra算法并记录每一步的状态"""
        # 初始化
        visited = set()
        previous = {}  # 记录每个节点的最短路径来源
        pq = [(0, 0)]  # (distance, node)
        processed_nodes = set()
        old_previous = {}  # 记录更新前的路径，用于标记变灰的边
        
        # 记录初始状态（源点0已在A集合中）
        processed_nodes.add(0)  # 源点0初始时已在A集合
        self.steps.append({
            'distances': self.distances.copy(),
            'current_node': None,
            'visited': visited.copy(),
            'processed': processed_nodes.copy(),
            'previous': previous.copy(),
            'old_previous': old_previous.copy(),
            'message': 'Initialization: Source 0 distance is 0, other vertices distance is ∞'
        })
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            processed_nodes.add(current_node)
            
            # 记录选择当前节点
            self.steps.append({
                'distances': self.distances.copy(),
                'current_node': current_node,
                'visited': visited.copy(),
                'processed': processed_nodes.copy(),
                'previous': previous.copy(),
                'old_previous': old_previous.copy(),
                'message': f'Select vertex {current_node} (distance: {current_dist})'
            })
            
            # 更新邻居节点
            for neighbor in self.G.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                edge_weight = self.G[current_node][neighbor]['weight']
                new_dist = current_dist + edge_weight
                
                # 记录检查边的过程
                self.steps.append({
                    'distances': self.distances.copy(),
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'processed': processed_nodes.copy(),
                    'checking_edge': (current_node, neighbor),
                    'edge_progress': 1.0,
                    'previous': previous.copy(),
                    'old_previous': old_previous.copy(),
                    'message': f'Check edge {current_node}→{neighbor} (weight: {edge_weight})'
                })
                
                if new_dist < self.distances[neighbor]:
                    old_dist = self.distances[neighbor]
                    # 记录旧路径（如果存在）
                    old_prev = previous.get(neighbor)
                    if old_prev is not None:
                        old_previous[neighbor] = old_prev
                    
                    # 立即更新距离和路径
                    self.distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))
                    
                    # 记录更新距离
                    self.steps.append({
                        'distances': self.distances.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'processed': processed_nodes.copy(),
                        'updated_node': neighbor,
                        'old_distance': old_dist,
                        'new_distance': new_dist,
                        'previous': previous.copy(),
                        'old_previous': old_previous.copy(),
                        'message': f'Update vertex {neighbor}: {old_dist if old_dist == float("inf") else old_dist} → {new_dist}, path {current_node}→{neighbor} added to shortest path'
                    })
                else:
                    # 记录不更新
                    self.steps.append({
                        'distances': self.distances.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'processed': processed_nodes.copy(),
                        'previous': previous.copy(),
                        'old_previous': old_previous.copy(),
                        'message': f'Vertex {neighbor} distance not improved (current: {self.distances[neighbor]}, new: {new_dist})'
                    })
        
        # 记录完成状态
        self.steps.append({
            'distances': self.distances.copy(),
            'current_node': None,
            'visited': visited.copy(),
            'processed': processed_nodes.copy(),
            'previous': previous.copy(),
            'old_previous': old_previous.copy(),
            'message': 'Algorithm complete! All shortest paths found'
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
        filename = f'dijkstra_step_{self.current_step:03d}.png'
        self.fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"已保存当前帧: {filename}")
    
    def export_all_frames(self, output_dir='dijkstra_frames'):
        """导出所有帧为图片序列，供PPT使用"""
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
        """绘制当前步骤的图"""
        if self.current_step >= len(self.steps):
            return
        
        step_data = self.steps[self.current_step]
        self.ax.clear()
        self.ax.set_facecolor('#f0f0f0')
        # 保持原图形的宽高比，使用原始坐标范围
        # 原图形xlim: -1到7（范围8），ylim: -2.5到2（范围4.5）
        # 使用原始坐标范围，图形区域为(-1,7)
        # 左侧留出表格空间(-5到-1)，右侧为图形区域(-1到7)
        # 不设置aspect='equal'，保持原图形的显示比例
        self.ax.set_xlim(-5, 7)
        self.ax.set_ylim(-2.5, 2)
        self.ax.axis('off')
        
        distances = step_data['distances']
        current_node = step_data.get('current_node')
        processed = step_data.get('processed', set())
        checking_edge = step_data.get('checking_edge')
        edge_progress = step_data.get('edge_progress', 1.0)
        updated_node = step_data.get('updated_node')
        previous = step_data.get('previous', {})
        old_previous = step_data.get('old_previous', {})
        
        # 绘制左侧算法演示表格
        self.draw_distance_table(distances, processed, current_node, updated_node)
        
        # 绘制边
        for edge in self.G.edges():
            u, v = edge
            weight = self.G[u][v]['weight']
            
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
                start_x = x1 + 0.15 * dx_norm
                start_y = y1 + 0.15 * dy_norm
                end_x = x2 - 0.15 * dx_norm
                end_y = y2 - 0.15 * dy_norm
                
                is_checking = checking_edge and checking_edge == (u, v)
                
                if is_checking:
                    # 正在检查的边，显示为红色
                    self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                   arrowprops=dict(arrowstyle='->', lw=6, 
                                                 color='red', alpha=0.9, zorder=3))
                else:
                    # 判断边是否在最短路径中
                    is_in_shortest_path = previous.get(v) == u
                    is_in_old_path = old_previous.get(v) == u
                    
                    if is_in_shortest_path:
                        edge_color = 'green'
                        edge_width = 4
                        linestyle = '-'
                        alpha = 0.9
                    elif is_in_old_path:
                        edge_color = 'gray'
                        edge_width = 3
                        linestyle = '--'
                        alpha = 0.4
                    else:
                        edge_color = 'gray'
                        edge_width = 3
                        linestyle = '-'
                        alpha = 0.5
                    
                    self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                   arrowprops=dict(arrowstyle='->', lw=edge_width, 
                                                 color=edge_color, alpha=alpha,
                                                 linestyle=linestyle))
                
                # 添加权重标签
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                edges_below = [(2, 5, 8), (1, 3, 1), (3, 5, 2)]
                is_below = (u, v, weight) in edges_below
                
                if is_below:
                    offset_x = -dy_norm * 0.25
                    offset_y = -abs(dx_norm) * 0.2 - 0.1
                else:
                    offset_x = -dy_norm * 0.2
                    offset_y = dx_norm * 0.2
                
                self.ax.text(mid_x + offset_x, mid_y + offset_y, str(weight),
                           fontsize=20, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                   edgecolor='gray', alpha=0.9, linewidth=3))
        
        # 绘制节点
        for node in self.G.nodes():
            x, y = self.pos[node]
            
            # 确定节点颜色
            if node == 0:
                node_color = 'orange'  # 源点
            elif node == current_node:
                node_color = 'yellow'  # 当前处理的节点
            elif node in processed:
                node_color = 'lightgreen'  # 已处理的节点
            elif updated_node == node:
                node_color = 'lightblue'  # 刚更新的节点
            else:
                node_color = 'white'  # 未处理的节点
            
            circle = plt.Circle((x, y), 0.3, color=node_color, 
                              ec='black', lw=4, zorder=3)
            self.ax.add_patch(circle)
            
            # 添加节点编号
            if node == 0:
                text_color = 'white'
            elif node_color in ['white', 'yellow', 'lightgreen', 'lightblue']:
                text_color = 'black'
            else:
                text_color = 'white'
            
            self.ax.text(x, y, str(node), fontsize=26, fontweight='bold',
                        ha='center', va='center', zorder=4, color=text_color)
            
            # 添加距离标签
            dist = distances[node]
            dist_text = '0' if node == 0 and dist == 0 else ('Inf' if dist == float('inf') else str(int(dist)))
            
            if node == 0:
                label_color = 'red'
            elif node == current_node or updated_node == node:
                label_color = 'blue'
            else:
                label_color = 'red'
            
            if node == 4 or node == 6:
                label_x = x + 0.4
                label_y = y - 0.4
                ha_align = 'left'
                va_align = 'bottom'
            else:
                label_x = x
                label_y = y - 0.6
                ha_align = 'center'
                va_align = 'center'
            
            self.ax.text(label_x, label_y, dist_text, fontsize=24, fontweight='bold',
                        ha=ha_align, va=va_align, color=label_color,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor=label_color, alpha=0.8, linewidth=3))
            
            # 源点标签
            if node == 0:
                self.ax.text(x - 0.5, y, 'source', fontsize=18, fontweight='bold',
                           ha='right', va='center', color='red', style='italic',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='red', alpha=0.7, linewidth=2.5))
        
        # 添加步骤信息（调整位置到图形中心，原位置是3）
        message = step_data.get('message', '')
        self.ax.text(3, -2.2, message, fontsize=20, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=1.0', facecolor='yellow', 
                            edgecolor='black', alpha=0.95, linewidth=4),
                    fontweight='bold')
        
        # 添加图例（移到左侧，避免被图形遮挡）
        legend_text = "Legend:\nOrange=Source | Yellow=Current | Green=Processed | Blue=Updated | Red edge=Checking"
        self.ax.text(-0.5, 1.9, legend_text, fontsize=16, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                            edgecolor='gray', alpha=0.9, linewidth=3),
                    fontweight='bold')
        
        self.fig.canvas.draw()
    
    def draw_distance_table(self, distances, processed, current_node, updated_node):
        """在左侧绘制2列8行的距离表格和集合信息"""
        # A集合：已处理的节点（processed）
        # B集合：未处理的节点（所有节点 - processed）
        all_nodes = sorted(self.G.nodes())
        set_A = processed
        set_B = set(all_nodes) - processed
        
        # 表格位置和尺寸
        table_x = -4.5
        table_y_top = 1.5
        row_height = 0.35
        col1_width = 1.2
        col2_width = 1.8
        total_width = col1_width + col2_width
        
        # 绘制表格边框和网格线
        num_rows = len(all_nodes) + 1  # 1行标题 + 7行数据
        
        # 绘制标题行
        title_y = table_y_top
        # 标题第一列
        title1_rect = plt.Rectangle((table_x, title_y - row_height), col1_width, row_height,
                                   facecolor='lightblue', edgecolor='black', 
                                   linewidth=2, zorder=1)
        self.ax.add_patch(title1_rect)
        self.ax.text(table_x + col1_width/2, title_y - row_height/2, 'Vertex', 
                    fontsize=16, fontweight='bold', ha='center', va='center', zorder=2)
        
        # 标题第二列
        title2_rect = plt.Rectangle((table_x + col1_width, title_y - row_height), col2_width, row_height,
                                   facecolor='lightblue', edgecolor='black', 
                                   linewidth=2, zorder=1)
        self.ax.add_patch(title2_rect)
        self.ax.text(table_x + col1_width + col2_width/2, title_y - row_height/2, 'Distance to Source', 
                    fontsize=16, fontweight='bold', ha='center', va='center', zorder=2)
        
        # 绘制数据行（7行数据：顶点0-6）
        for i, node in enumerate(all_nodes):
            row_y = title_y - row_height * (i + 1)  # 紧接标题下方
            
            # 判断是否高亮（在A集合中）
            is_in_set_A = node in set_A
            is_current = node == current_node
            is_updated = node == updated_node
            
            # 行背景颜色
            if is_in_set_A:
                row_color = 'lightgreen'  # A集合中的节点用绿色高亮
                edge_color = 'green'
                edge_width = 2
            elif is_current:
                row_color = 'yellow'  # 当前节点用黄色
                edge_color = 'orange'
                edge_width = 2
            elif is_updated:
                row_color = 'lightblue'  # 刚更新的节点用蓝色
                edge_color = 'blue'
                edge_width = 2
            else:
                row_color = 'white'
                edge_color = 'black'
                edge_width = 1
            
            # 绘制第一列（顶点编号）
            rect1 = plt.Rectangle((table_x, row_y - row_height), col1_width, row_height,
                                 facecolor=row_color, edgecolor=edge_color, 
                                 linewidth=edge_width, zorder=1)
            self.ax.add_patch(rect1)
            self.ax.text(table_x + col1_width/2, row_y - row_height/2, str(node), 
                        fontsize=18, fontweight='bold', ha='center', va='center', zorder=2)
            
            # 绘制第二列（到达源点的距离）
            rect2 = plt.Rectangle((table_x + col1_width, row_y - row_height), col2_width, row_height,
                                 facecolor=row_color, edgecolor=edge_color, 
                                 linewidth=edge_width, zorder=1)
            self.ax.add_patch(rect2)
            
            # 距离值
            dist = distances[node]
            if dist == float('inf'):
                dist_text = '∞'
            else:
                dist_text = str(int(dist))
            
            self.ax.text(table_x + col1_width + col2_width/2, row_y - row_height/2, dist_text, 
                        fontsize=18, fontweight='bold', ha='center', va='center', zorder=2)
        
        # 绘制表格外边框
        table_bottom = title_y - row_height * (num_rows)
        border_rect = plt.Rectangle((table_x, table_bottom), total_width, row_height * num_rows,
                                   facecolor='none', edgecolor='black', 
                                   linewidth=3, zorder=0)
        self.ax.add_patch(border_rect)
        
        # 绘制列分隔线
        for i in range(num_rows):
            y_pos = title_y - row_height * i
            self.ax.plot([table_x + col1_width, table_x + col1_width], 
                        [y_pos - row_height, y_pos],
                        color='black', linewidth=2, zorder=2)
        
        # 绘制集合信息（在表格下方）
        set_y = table_bottom - 0.4
        
        # A集合
        set_A_text = f'Set A: {{{", ".join(map(str, sorted(set_A)))}}}'
        self.ax.text(table_x + total_width/2, set_y, set_A_text, fontsize=16, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', 
                            edgecolor='green', alpha=0.9, linewidth=2))
        
        # B集合
        set_B_text = f'Set B: {{{", ".join(map(str, sorted(set_B)))}}}'
        set_B_y = set_y - 0.4
        self.ax.text(table_x + total_width/2, set_B_y, set_B_text, fontsize=16, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                            edgecolor='orange', alpha=0.9, linewidth=2))
    
    def show(self):
        """显示交互式窗口"""
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # 创建交互式可视化对象
    viz = DijkstraInteractive()
    
    # 可选：导出所有帧为图片序列（取消注释以使用）
    # viz.export_all_frames('dijkstra_frames')
    
    # 显示交互式窗口
    viz.show()

