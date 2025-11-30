import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from collections import defaultdict
import heapq
import matplotlib
from matplotlib.colors import to_rgb

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
        
        # 创建图形（增大尺寸以提高清晰度）
        self.fig, self.ax = plt.subplots(figsize=(16, 10), dpi=200)
        self.ax.set_facecolor('#f0f0f0')
        
    def run_dijkstra(self):
        """执行Dijkstra算法并记录每一步的状态"""
        # 初始化
        visited = set()
        previous = {}  # 记录每个节点的最短路径来源
        pq = [(0, 0)]  # (distance, node)
        processed_nodes = set()
        old_previous = {}  # 记录更新前的路径，用于标记变灰的边
        
        # 记录初始状态
        self.steps.append({
            'distances': self.distances.copy(),
            'current_node': None,
            'visited': visited.copy(),
            'processed': processed_nodes.copy(),
            'previous': previous.copy(),  # 记录当前最短路径
            'message': '初始化：源点0的距离为0，其他节点距离为∞'
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
                'message': f'选择节点 {current_node}（距离：{current_dist}）'
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
                    'message': f'开始检查边 {current_node}→{neighbor}（权重：{edge_weight}）'
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
                        'message': f'检查边 {current_node}→{neighbor}（权重：{edge_weight}）'
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
                    'message': f'检查边 {current_node}→{neighbor}（权重：{edge_weight}）'
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
                        'message': f'更新节点 {neighbor}：{old_dist if old_dist == float("inf") else old_dist} → {new_dist}，路径 {current_node}→{neighbor} 变绿色'
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
                        'message': f'节点 {neighbor} 距离未改善（当前：{self.distances[neighbor]}，新：{new_dist}）'
                    })
        
        # 记录完成状态
        self.steps.append({
            'distances': self.distances.copy(),
            'current_node': None,
            'visited': visited.copy(),
            'processed': processed_nodes.copy(),
            'previous': previous.copy(),  # 记录最终最短路径
            'message': '算法完成！所有最短路径已找到'
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
        
        return transition
    
    def draw_graph(self, step_data):
        """绘制当前步骤的图"""
        self.ax.clear()
        self.ax.set_facecolor('#f0f0f0')
        self.ax.set_xlim(-1, 7)
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
        legend_text = "图例：\n橙色=源点 | 黄色=当前节点 | 绿色=已处理 | 蓝色=刚更新 | 红色边=正在检查"
        self.ax.text(0.2, 1.9, legend_text, fontsize=16, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                            edgecolor='gray', alpha=0.9, linewidth=3),
                    fontweight='bold')
    
    def animate(self, frame):
        """动画函数"""
        if frame < len(self.steps):
            self.draw_graph(self.steps[frame])
        return []
    
    def create_animation(self, output_file='dijkstra_animation.gif', interval=100, dpi=200):
        """创建动画并保存为GIF
        
        参数:
            output_file: 输出文件名
            interval: 每帧间隔时间（毫秒）
            dpi: 图像分辨率
        """
        # 启用抗锯齿
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 12
        
        # 保存为GIF（使用更高的质量设置）
        total_frames = len(self.steps)
        print(f"正在生成动画，共 {total_frames} 帧...")
        print(f"这可能需要一些时间，请稍候...")
        print(f"提示：生成高分辨率动画需要较长时间，请耐心等待...")
        
        # 创建动画对象
        anim = animation.FuncAnimation(self.fig, self.animate, 
                                      frames=total_frames,
                                      interval=interval, 
                                      repeat=True, 
                                      blit=False)
        
        # 保存为GIF，使用更高的质量
        fps = 1000 / interval if interval > 0 else 10
        anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
        
        print(f"\n✓ 动画已成功保存为: {output_file}")
        print(f"  - 总帧数: {len(self.steps)}")
        print(f"  - 分辨率: {dpi} DPI")
        print(f"  - 每帧间隔: {interval}ms")
        
        return anim

if __name__ == '__main__':
    print("=" * 60)
    print("Dijkstra算法动态演示生成器")
    print("=" * 60)
    
    # 创建可视化对象
    print("\n正在初始化图结构...")
    viz = DijkstraVisualization()
    
    # 生成动画（提高DPI和调整间隔）
    print("\n开始生成动画...")
    anim = viz.create_animation('dijkstra_demo.gif', interval=100, dpi=200)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    
    # 可选：显示动画窗口（注释掉以避免阻塞）
    # plt.tight_layout()
    # plt.show()

