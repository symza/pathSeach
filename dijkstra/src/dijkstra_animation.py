import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from collections import defaultdict
import heapq
import matplotlib
from matplotlib.colors import to_rgb
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
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def interpolate_color(color1, color2, alpha):
    """在两个颜色之间插值
    alpha: 0.0 表示 color1, 1.0 表示 color2
    """
    rgb1 = np.array(to_rgb(color1))
    rgb2 = np.array(to_rgb(color2))
    rgb = rgb1 * (1 - alpha) + rgb2 * alpha
    return tuple(rgb)

class DijkstraVisualization:
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
        
        # 设置节点位置（使用spring布局，但固定位置）
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
        
        # 扩展步骤以增加帧数到500
        self.expand_steps_to_500()
        
        # 创建图形（增大尺寸以提高清晰度，左侧留出表格空间）
        self.fig, self.ax = plt.subplots(figsize=(22, 10), dpi=200)
        self.ax.set_facecolor('#f0f0f0')
        
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
            'previous': previous.copy(),  # 记录当前最短路径
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
                'previous': previous.copy(),  # 记录当前最短路径
                'old_previous': old_previous.copy(),
                'message': f'Select vertex {current_node} (distance: {current_dist})'
            })
            
            # 更新邻居节点
            for neighbor in self.G.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                edge_weight = self.G[current_node][neighbor]['weight']
                new_dist = current_dist + edge_weight
                
                # 记录检查边的过程（添加多个帧来显示线条逐渐延伸）
                # 先添加开始检查的帧
                self.steps.append({
                    'distances': self.distances.copy(),
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'processed': processed_nodes.copy(),
                    'checking_edge': (current_node, neighbor),
                    'edge_progress': 0.0,  # 线条延伸进度
                    'previous': previous.copy(),  # 记录当前最短路径
                    'old_previous': old_previous.copy(),
                    'message': f'Check edge {current_node}→{neighbor} (weight: {edge_weight})'
                })
                
                # 添加线条逐渐延伸的中间帧（15帧使动画更流畅）
                # 在检查过程中，保持previous不变（搜索过程中路径不变）
                current_previous = previous.copy()
                for progress in [i/15.0 for i in range(1, 15)]:  # 0.067, 0.133, ..., 0.933
                    self.steps.append({
                        'distances': self.distances.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'processed': processed_nodes.copy(),
                        'checking_edge': (current_node, neighbor),
                        'edge_progress': progress,
                        'previous': current_previous.copy(),  # 搜索过程中路径不变
                        'old_previous': old_previous.copy(),
                        'message': f'Check edge {current_node}→{neighbor} (weight: {edge_weight})'
                    })
                
                # 添加检查完成的帧
                self.steps.append({
                    'distances': self.distances.copy(),
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'processed': processed_nodes.copy(),
                    'checking_edge': (current_node, neighbor),
                    'edge_progress': 1.0,  # 线条完全延伸
                    'previous': previous.copy(),  # 记录当前最短路径
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
                    
                    # 立即记录更新距离，此时边应该变绿色（在最短路径中）
                    step_data = {
                        'distances': self.distances.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'processed': processed_nodes.copy(),
                        'updated_node': neighbor,
                        'old_distance': old_dist,
                        'new_distance': new_dist,
                        'previous': previous.copy(),  # 记录当前最短路径（包含刚更新的边）
                        'old_previous': old_previous.copy() if old_prev is not None else {},  # 记录被替换的旧路径
                        'message': f'Update vertex {neighbor}: {old_dist if old_dist == float("inf") else old_dist} → {new_dist}, path {current_node}→{neighbor} added to shortest path'
                    }
                    self.steps.append(step_data)
                else:
                    # 记录不更新（距离没有改善，边检查完成）
                    self.steps.append({
                        'distances': self.distances.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'processed': processed_nodes.copy(),
                        'previous': previous.copy(),  # 记录当前最短路径
                        'old_previous': old_previous.copy(),
                        'message': f'Vertex {neighbor} distance not improved (current: {self.distances[neighbor]}, new: {new_dist})'
                    })
        
        # 记录完成状态
        self.steps.append({
            'distances': self.distances.copy(),
            'current_node': None,
            'visited': visited.copy(),
            'processed': processed_nodes.copy(),
            'previous': previous.copy(),  # 记录最终最短路径
            'old_previous': old_previous.copy(),
            'message': 'Algorithm complete! All shortest paths found'
        })
    
    def expand_steps_to_500(self):
        """扩展步骤以增加帧数到500，添加过渡帧和重复帧"""
        expanded_steps = []
        target_frames = 500
        base_steps = len(self.steps)
        
        if base_steps == 0:
            return
        
        # 为重要步骤分配更多帧数
        # 重要步骤：初始化、选择节点、更新距离、完成
        important_indices = {0, base_steps - 1}  # 初始化和完成
        
        # 识别重要步骤（选择节点、更新距离）
        for i, step in enumerate(self.steps):
            if step.get('current_node') is not None and 'checking_edge' not in step:
                important_indices.add(i)
            if 'updated_node' in step:
                important_indices.add(i)
        
        # 计算基础帧数分配
        base_frames_per_step = max(3, (target_frames - len(important_indices) * 10) // base_steps)
        
        for i, step in enumerate(self.steps):
            # 重要步骤分配更多帧
            if i in important_indices:
                frames_for_this_step = base_frames_per_step + 5
            else:
                frames_for_this_step = base_frames_per_step
            
            # 添加重复帧（在重复帧之间也添加微小的过渡）
            for repeat in range(frames_for_this_step):
                if repeat == 0:
                    expanded_steps.append(step.copy())
                else:
                    # 在重复帧中添加微小的过渡效果（保持状态但添加轻微变化）
                    expanded_steps.append(step.copy())
            
            # 在步骤之间添加过渡帧（平滑过渡）
            if i < len(self.steps) - 1:
                next_step = self.steps[i + 1]
                # 添加更多过渡帧以实现平滑颜色过渡（5-8帧）
                num_transitions = 6
                for transition in range(num_transitions):
                    alpha = (transition + 1) / (num_transitions + 1.0)
                    transition_step = self.create_transition_step(step, next_step, alpha)
                    expanded_steps.append(transition_step)
        
        # 调整到精确的500帧
        current_frames = len(expanded_steps)
        if current_frames < target_frames:
            # 在最后添加重复帧
            while len(expanded_steps) < target_frames:
                expanded_steps.append(self.steps[-1].copy())
        elif current_frames > target_frames:
            # 均匀采样到500帧
            indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
            expanded_steps = [expanded_steps[i] for i in indices]
        
        self.steps = expanded_steps[:target_frames]
    
    def create_transition_step(self, step1, step2, alpha):
        """创建两个步骤之间的过渡步骤"""
        transition = step1.copy()
        
        # 记录过渡alpha值，用于颜色插值
        transition['transition_alpha'] = alpha
        transition['prev_step'] = step1
        transition['next_step'] = step2
        
        # 处理边的进度过渡
        if 'checking_edge' in step2 and 'edge_progress' in step2:
            transition['checking_edge'] = step2['checking_edge']
            # 如果step1也有相同的边，则插值进度
            if 'checking_edge' in step1 and step1['checking_edge'] == step2['checking_edge']:
                progress1 = step1.get('edge_progress', 0.0)
                progress2 = step2.get('edge_progress', 1.0)
                transition['edge_progress'] = progress1 * (1 - alpha) + progress2 * alpha
            else:
                # 新开始检查的边，从0开始
                transition['edge_progress'] = step2.get('edge_progress', 0.0) * alpha
        elif 'checking_edge' in step1 and 'edge_progress' in step1:
            # 保持前一步的边和进度
            transition['checking_edge'] = step1['checking_edge']
            transition['edge_progress'] = step1.get('edge_progress', 1.0)
        
        # 如果步骤2有新的更新，在过渡中显示
        if 'updated_node' in step2 and 'updated_node' not in step1:
            transition['updated_node'] = step2.get('updated_node')
            # 距离也进行插值过渡
            if 'distances' in step2:
                transition['distances'] = {}
                for node in step2['distances']:
                    d1 = step1['distances'].get(node, float('inf'))
                    d2 = step2['distances'].get(node, float('inf'))
                    if d1 == float('inf') and d2 == float('inf'):
                        transition['distances'][node] = float('inf')
                    elif d1 == float('inf'):
                        transition['distances'][node] = d2
                    elif d2 == float('inf'):
                        transition['distances'][node] = d1
                    else:
                        transition['distances'][node] = d1 * (1 - alpha) + d2 * alpha
            transition['message'] = step2.get('message', '')
        
        # 更新当前节点（在过渡中逐渐变化）
        if 'current_node' in step2:
            transition['current_node'] = step2.get('current_node')
        
        # 更新已处理的节点
        if 'processed' in step2:
            transition['processed'] = step2['processed'].copy()
        
        # 确保包含 old_previous 字段
        if 'old_previous' not in transition:
            transition['old_previous'] = step1.get('old_previous', {})
        
        return transition
    
    def draw_graph(self, step_data):
        """绘制当前步骤的图"""
        self.ax.clear()
        self.ax.set_facecolor('#f0f0f0')
        # 调整坐标范围，左侧留出表格空间
        self.ax.set_xlim(-5, 7)
        self.ax.set_ylim(-2.5, 2)  # 向下扩展以容纳下移的信息框
        self.ax.axis('off')
        
        distances = step_data['distances']
        current_node = step_data.get('current_node')
        processed = step_data.get('processed', set())
        checking_edge = step_data.get('checking_edge')
        edge_progress = step_data.get('edge_progress', 1.0)  # 线条延伸进度，默认1.0（完整）
        updated_node = step_data.get('updated_node')
        transition_alpha = step_data.get('transition_alpha', 0.0)
        prev_step = step_data.get('prev_step')
        next_step = step_data.get('next_step')
        previous = step_data.get('previous', {})  # 最短路径来源
        old_previous = step_data.get('old_previous', {})  # 被替换的旧路径
        
        # 绘制左侧算法演示表格
        self.draw_distance_table(distances, processed, current_node, updated_node)
        
        # 绘制边
        for edge in self.G.edges():
            u, v = edge
            weight = self.G[u][v]['weight']
            
            # 计算边的起点和终点
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            
            # 计算箭头位置（稍微偏移以避免重叠）
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
                # 箭头起点和终点（考虑节点半径）
                start_x = x1 + 0.15 * dx_norm
                start_y = y1 + 0.15 * dy_norm
                end_x = x2 - 0.15 * dx_norm
                end_y = y2 - 0.15 * dy_norm
                
                # 确定是否正在检查这条边
                is_checking = checking_edge and checking_edge == (u, v)
                
                # 如果正在检查这条边，绘制逐渐延伸的红色线
                if is_checking:
                    # 计算部分线段的终点
                    partial_length_x = (end_x - start_x) * edge_progress
                    partial_length_y = (end_y - start_y) * edge_progress
                    partial_end_x = start_x + partial_length_x
                    partial_end_y = start_y + partial_length_y
                    
                    # 先绘制完整的背景线（灰色，较淡）
                    self.ax.plot([start_x, end_x], [start_y, end_y],
                               color='gray', linewidth=3, alpha=0.3, linestyle='--', zorder=1)
                    
                    # 绘制逐渐延伸的红色高亮线
                    if edge_progress < 1.0:
                        # 部分线段，不绘制箭头，使用圆形端点
                        self.ax.plot([start_x, partial_end_x], [start_y, partial_end_y],
                                   color='red', linewidth=6, alpha=0.9, zorder=3, solid_capstyle='round')
                        # 在端点添加一个小圆点
                        self.ax.plot(partial_end_x, partial_end_y, 'ro', 
                                   markersize=12, zorder=4, alpha=0.9)
                    else:
                        # 完整线段，绘制箭头
                        self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                       arrowprops=dict(arrowstyle='->', lw=6, 
                                                     color='red', alpha=0.9, zorder=3))
                else:
                    # 不在检查中，绘制正常的边
                    # 判断边是否在最短路径中（找到最短路径的边立即变绿）
                    is_in_shortest_path = previous.get(v) == u
                    # 判断边是否在旧路径中（被更短路径替代）
                    is_in_old_path = old_previous.get(v) == u
                    
                    if is_in_shortest_path:
                        # 在最短路径中，显示为绿色（找到最短路径时立即变绿）
                        edge_color = 'green'
                        edge_width = 4
                        linestyle = '-'
                        alpha = 0.9
                    elif is_in_old_path:
                        # 在旧路径中（被替代），显示为灰色虚线
                        edge_color = 'gray'
                        edge_width = 3
                        linestyle = '--'
                        alpha = 0.4
                    else:
                        # 其他边：搜索过程中保持灰色
                        edge_color = 'gray'
                        edge_width = 3
                        linestyle = '-'
                        alpha = 0.5
                    
                    self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                   arrowprops=dict(arrowstyle='->', lw=edge_width, 
                                                 color=edge_color, alpha=alpha,
                                                 linestyle=linestyle))
                
                # 添加权重标签（增大字体，加粗边框）
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # 判断是否需要将标签放在线下方
                # 需要放在下方的边：(2,5)权值8, (1,3)权值1, (3,5)权值2
                edges_below = [(2, 5, 8), (1, 3, 1), (3, 5, 2)]
                is_below = (u, v, weight) in edges_below
                
                if is_below:
                    # 放在线下方：垂直于线方向向下偏移，紧挨着线
                    offset_x = -dy_norm * 0.25
                    offset_y = -abs(dx_norm) * 0.2 - 0.1  # 进一步减小偏移量，让标签更紧挨着线
                else:
                    # 默认位置：偏移标签位置以避免重叠
                    offset_x = -dy_norm * 0.2
                    offset_y = dx_norm * 0.2
                
                self.ax.text(mid_x + offset_x, mid_y + offset_y, str(weight),
                           fontsize=20, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                   edgecolor='gray', alpha=0.9, linewidth=3))
        
        # 绘制节点
        for node in self.G.nodes():
            x, y = self.pos[node]
            
            # 确定节点颜色（支持过渡）
            if transition_alpha > 0 and prev_step and next_step:
                # 计算前一步的节点颜色
                prev_current = prev_step.get('current_node')
                prev_processed = prev_step.get('processed', set())
                prev_updated = prev_step.get('updated_node')
                
                if node == 0:
                    prev_color = 'orange'
                elif node == prev_current:
                    prev_color = 'yellow'
                elif node in prev_processed:
                    prev_color = 'lightgreen'
                elif prev_updated == node:
                    prev_color = 'lightblue'
                else:
                    prev_color = 'white'
                
                # 计算下一步的节点颜色
                next_current = next_step.get('current_node') if next_step else current_node
                next_processed = next_step.get('processed', set()) if next_step else processed
                next_updated = next_step.get('updated_node') if next_step else updated_node
                
                if node == 0:
                    next_color = 'orange'
                elif node == next_current:
                    next_color = 'yellow'
                elif node in next_processed:
                    next_color = 'lightgreen'
                elif next_updated == node:
                    next_color = 'lightblue'
                else:
                    next_color = 'white'
                
                # 颜色插值
                node_color = interpolate_color(prev_color, next_color, transition_alpha)
            else:
                # 正常颜色
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
            
            # 绘制节点圆圈（处理颜色格式）
            if isinstance(node_color, tuple):
                node_color_final = node_color
            else:
                node_color_final = node_color
            
            circle = plt.Circle((x, y), 0.3, color=node_color_final, 
                              ec='black', lw=4, zorder=3)
            self.ax.add_patch(circle)
            
            # 添加节点编号（增大字体，提高清晰度，无背景框）
            # 根据节点颜色选择文字颜色，确保可读性
            # 源节点（橙色）固定使用白色文字
            if node == 0:
                text_color = 'white'
            elif isinstance(node_color, tuple):
                # RGB颜色，计算亮度来决定文字颜色
                brightness = sum(node_color) / 3.0
                text_color = 'black' if brightness > 0.5 else 'white'
            else:
                # 根据颜色名称选择文字颜色
                if node_color in ['white', 'yellow', 'lightgreen', 'lightblue']:
                    text_color = 'black'
                else:
                    text_color = 'white'
            
            self.ax.text(x, y, str(node), fontsize=26, fontweight='bold',
                        ha='center', va='center', zorder=4, color=text_color)
            
            # 添加距离标签（增大字体）
            dist = distances[node]
            dist_text = '0' if node == 0 and dist == 0 else ('Inf' if dist == float('inf') else str(int(dist)))
            
            # 确定标签颜色
            if node == 0:
                label_color = 'red'
            elif node == current_node or updated_node == node:
                label_color = 'blue'
            else:
                label_color = 'red'
            
            # 根据节点位置调整标签位置
            if node == 4 or node == 6:
                # 节点4和6的Inf文字放在右下位置
                label_x = x + 0.4
                label_y = y - 0.4
                ha_align = 'left'
                va_align = 'bottom'
            else:
                # 其他节点放在下方居中
                label_x = x
                label_y = y - 0.6
                ha_align = 'center'
                va_align = 'center'
            
            self.ax.text(label_x, label_y, dist_text, fontsize=24, fontweight='bold',
                        ha=ha_align, va=va_align, color=label_color,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor=label_color, alpha=0.8, linewidth=3))
            
            # 如果是源点，添加"source"标签在左边位置
            if node == 0:
                self.ax.text(x - 0.5, y, 'source', fontsize=18, fontweight='bold',
                           ha='right', va='center', color='red', style='italic',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='red', alpha=0.7, linewidth=2.5))
        
        # 添加步骤信息（增大字体，提高清晰度，加粗边框，向下移动避免遮挡）
        message = step_data.get('message', '')
        self.ax.text(3, -2.2, message, fontsize=20, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=1.0', facecolor='yellow', 
                            edgecolor='black', alpha=0.95, linewidth=4),
                    fontweight='bold')
        
        # 添加图例（增大字体，加粗边框，向上移动）
        legend_text = "Legend:\nOrange=Source | Yellow=Current | Green=Processed | Blue=Updated | Red edge=Checking"
        self.ax.text(-0.5, 1.9, legend_text, fontsize=16, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                            edgecolor='gray', alpha=0.9, linewidth=3),
                    fontweight='bold')
    
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
    
    def animate(self, frame):
        """动画函数"""
        if frame < len(self.steps):
            self.draw_graph(self.steps[frame])
        return []
    
    def create_animation(self, output_file=None, interval=None, dpi=200):
        """创建动画并保存为GIF
        
        参数:
            output_file: 输出文件名，如果为None则保存到images文件夹
            interval: 每帧间隔时间（毫秒），如果为None则自动计算以限制在10秒内
            dpi: 图像分辨率
        """
        # 设置默认输出文件路径
        if output_file is None:
            output_file = os.path.join(IMAGES_DIR, 'dijkstra_animation.gif')
        elif not os.path.isabs(output_file):
            # 如果是相对路径，则保存到images文件夹
            output_file = os.path.join(IMAGES_DIR, output_file)
        
        # 启用抗锯齿
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 12
        
        # 保存为GIF（使用更高的质量设置）
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
        
        # 创建动画对象
        anim = animation.FuncAnimation(self.fig, self.animate, 
                                      frames=total_frames,
                                      interval=interval, 
                                      repeat=True, 
                                      blit=False)
        
        # 保存为GIF，使用更高的质量
        fps = 1000 / interval if interval > 0 else 10
        anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
        
        print(f"\n✓ Animation successfully saved as: {output_file}")
        print(f"  - Total frames: {len(self.steps)}")
        print(f"  - Resolution: {dpi} DPI")
        print(f"  - Frame interval: {interval}ms")
        print(f"  - Total duration: {actual_total_time:.1f} seconds")
        
        return anim

if __name__ == '__main__':
    print("=" * 60)
    print("Dijkstra Algorithm Animation Generator")
    print("=" * 60)
    
    # 创建可视化对象
    print("\nInitializing graph structure...")
    viz = DijkstraVisualization()
    
    # 生成动画（自动计算间隔以限制在10秒内）
    print("\nStarting animation generation...")
    anim = viz.create_animation(None, interval=None, dpi=200)  # None表示使用默认路径
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    
    # 可选：显示动画窗口（注释掉以避免阻塞）
    # plt.tight_layout()
    # plt.show()

