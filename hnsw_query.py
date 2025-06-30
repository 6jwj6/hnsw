import math
import random
import heapq
import time
import sys
import os
import copy

# --- 全局常量和变量 ---
MINX = -100
MAXX = 100

D = 4
M = 6
MMAX0 = 2 * M
MMAX = M
N = 10
EF_CONSTRUCTION = 5
K = 1

# --- 数据容器分离 ---
node = []      # 存储主数据集, 索引 0-N-1
querynode = [] # 存储查询点, 索引 0-nq-1

edge = [[]]
ite = [0]

# --- HNSW 算法状态 ---
L = 0
EP = 1
nodecnt = 0
ML = 1 / math.log(M)

# --- 辅助函数 ---
def gendata(num=N, dim=D):
    """生成随机数据并写入 data.txt"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "data.txt")
    print("开始生成数据...", file=sys.stderr)
    with open(output_path, "w") as f:
        for _ in range(num):
            line = ' '.join(str(random.randint(MINX, MAXX)) for _ in range(dim))
            f.write(line + "\n")
    print("完成写入 data.txt", file=sys.stderr)

# --- 修改开始 (1/6): 创建两个版本的 distance 函数 ---
def distance_node_to_node(idx1, idx2):
    """计算主数据集中两个节点之间的距离"""
    vec1 = node[idx1]
    vec2 = node[idx2]
    res_sq = sum((vec1[i] - vec2[i]) ** 2 for i in range(D))
    return res_sq #math.sqrt(res_sq)

def distance_query_to_node(q_idx, node_idx):
    """计算一个查询点和一个主数据集节点之间的距离"""
    vec1 = querynode[q_idx]
    vec2 = node[node_idx]
    res_sq = sum((vec1[i] - vec2[i]) ** 2 for i in range(D))
    return res_sq #math.sqrt(res_sq)
# --- 修改结束 (1/6) ---

def nxt(W):
    return [i + 1 for i in W]

# --- HNSW 核心算法 ---

# --- 修改开始 (2/6): search_layer 现在专用于插入流程 ---
def search_layer(q_idx, ep, ef, lc):
    """
    在指定层 (lc) 上搜索最近的 ef 个邻居.
    此版本专用于插入流程，q_idx 是 node 列表的索引。
    """
    v = set(ep)
    # 调用 node-to-node 距离函数
    C = [(distance_node_to_node(q_idx, ite[x]), x) for x in ep]
    heapq.heapify(C)
    W = [(-distance_node_to_node(q_idx, ite[x]), x) for x in ep]
    heapq.heapify(W)

    while C:
        dist_c, c_node_idx = heapq.heappop(C)
        if dist_c > -W[0][0]:
            break
        for e_node_idx in edge[c_node_idx]:
            if e_node_idx not in v:
                v.add(e_node_idx)
                # 调用 node-to-node 距离函数
                dist_e = distance_node_to_node(q_idx, ite[e_node_idx])
                if len(W) < ef or dist_e < -W[0][0]:
                    heapq.heappush(C, (dist_e, e_node_idx))
                    heapq.heappush(W, (-dist_e, e_node_idx))
                    if len(W) > ef:
                        heapq.heappop(W)
    return [item[1] for item in W]
    # return [item[1] for item in sorted(W, key=lambda x: x[0], reverse=True)]
# --- 修改结束 (2/6) ---

# --- 修改开始 (3/6): 为查询流程创建 search_layer_for_query ---
def search_layer_for_query(q_idx, ep, ef, lc):
    """
    在指定层 (lc) 上搜索最近的 ef 个邻居.
    这是 search_layer 的一个副本，但专用于查询流程。
    q_idx 是 querynode 列表的索引。
    """
    v = set(ep)
    # 调用 query-to-node 距离函数
    C = [(distance_query_to_node(q_idx, ite[x]), x) for x in ep]
    heapq.heapify(C)
    W = [(-distance_query_to_node(q_idx, ite[x]), x) for x in ep]
    heapq.heapify(W)

    while C:
        dist_c, c_node_idx = heapq.heappop(C)
        if dist_c > -W[0][0]:
            break
        for e_node_idx in edge[c_node_idx]:
            if e_node_idx not in v:
                v.add(e_node_idx)
                # 调用 query-to-node 距离函数
                dist_e = distance_query_to_node(q_idx, ite[e_node_idx])
                if len(W) < ef or dist_e < -W[0][0]:
                    heapq.heappush(C, (dist_e, e_node_idx))
                    heapq.heappush(W, (-dist_e, e_node_idx))
                    if len(W) > ef:
                        heapq.heappop(W)
    return [item[1] for item in sorted(W, key=lambda x: x[0], reverse=True)]

def select_neighbors_simple(q_idx, C, M_conn, lc):
    R = [] 
    W = [] # 工作小顶堆 (距离, 图索引)
    for x in C:
        heapq.heappush(W, (distance_node_to_node(q_idx, ite[x]), x))
    while W and len(R) < M_conn:
        R.append(heapq.heappop(W)[1])
    return R

# select_neighbors_heuristic 只被 insert 调用，所以它也只处理 node 内部的距离
def select_neighbors_heuristic(q_idx, C, M_conn, lc, extend_candidates=False, keep_pruned_connections=True):
    R, W, Wd = [], [], []
    for x in C:
        heapq.heappush(W, (distance_node_to_node(q_idx, ite[x]), x))
    # ... (其余逻辑不变)
    if extend_candidates:
        st = set(C)
        for e in C:
            for e_adj in edge[e]:
                if e_adj not in st:
                    st.add(e_adj)
                    heapq.heappush(W, (distance_node_to_node(q_idx, ite[e_adj]), e_adj))
    while W and len(R) < M_conn:
        dist_e, e_node_idx = heapq.heappop(W)
        if not R or dist_e < R[0][0]:
            heapq.heappush(R, (dist_e, e_node_idx))
        else:
            heapq.heappush(Wd, (dist_e, e_node_idx))
    if keep_pruned_connections:
        while Wd and len(R) < M_conn:
            heapq.heappush(R, heapq.heappop(Wd))
    return [item[1] for item in R]

# insert 函数逻辑不变，因为它总是处理主数据集内部的节点
def insert(q_idx, M_param, Mmax_param, efConstruction_param, mL_param):
    global L, EP, nodecnt
    current_ep = [EP] if nodecnt != 0 else []
    l = math.floor(-math.log(random.uniform(0.0, 1.0)) * mL_param)
    if l > L:
        EP = nodecnt + 1
        for _ in range(l, L, -1):
            nodecnt += 1
            ite.append(q_idx)
            edge.append([])
    for lc in range(L, l, -1):
        W = search_layer(q_idx, current_ep, 1, lc)
        if not W: break
        assert len(W) == 1
        current_ep = nxt(W)
    for lc in range(min(L, l), -1, -1):
        m_max_lc = MMAX0 if lc == 0 else Mmax_param
        nodecnt += 1
        ite.append(q_idx)
        W = search_layer(q_idx, current_ep, efConstruction_param, lc)
        if not W:
            edge.append([])
            continue
        neighbors = select_neighbors_simple(q_idx, W, M_param, lc)
        edge.append(neighbors)
        new_node_graph_idx = nodecnt
        for e_node_idx in neighbors:
            edge[e_node_idx].append(new_node_graph_idx)
            if len(edge[e_node_idx]) > m_max_lc:
                e_conn = edge[e_node_idx]
                e_new_conn = select_neighbors_simple(ite[e_node_idx], e_conn, m_max_lc, lc)
                edge[e_node_idx] = e_new_conn
        current_ep = nxt(W)
    if l > L:
        L = l

# --- 修改开始 (4/6): 修改 k_nn_search 以调用查询专用函数 ---
def k_nn_search(q_idx, k_param, ef_param):
    """
    在 HNSW 图中搜索 q_idx 的 K 个最近邻。
    q_idx 是 querynode 列表的索引。
    """
    W = []
    ep = [EP]
    
    for lc in range(L, 0, -1):
        # 调用查询专用版本
        W = search_layer_for_query(q_idx, ep, 1, lc)
        assert len(W) == 1
        ep = nxt(W)

    # 调用查询专用版本
    W = search_layer_for_query(q_idx, ep, ef_param, 0)
    
    result_heap = []
    for x_node_idx in W:
        dist = distance_query_to_node(q_idx, ite[x_node_idx])
        heapq.heappush(result_heap, (-dist, ite[x_node_idx]))
        if len(result_heap) > k_param:
            heapq.heappop(result_heap)
            
    return [item[1] for item in sorted(result_heap, key=lambda x: x[0], reverse=True)]
# --- 修改结束 (4/6) ---

# --- 初始化和主程序 ---
def init():
    """从文件加载数据"""
    global node
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data.txt")
    print("开始读取数据...", file=sys.stderr)
    node = []
    # 保持 1-based 索引的兼容性，在 node[0] 处插入一个占位符
    node.append([0] * D) 
    try:
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if i >= N: break
                node.append([int(x) for x in line.strip().split()])
    except FileNotFoundError:
        print(f"错误: 在路径 {data_path} 未找到 data.txt。", file=sys.stderr)
        exit(1)
    print("完成读取 data.txt", file=sys.stderr)
    if len(node) > 4:
        for i in range(1, 4):
            print(f"节点 {i}: {node[i]}", file=sys.stderr)

# bruteforce 函数现在也专用于查询
def bruteforce_for_query(q_idx, print_flag = True):
    """暴力搜索查询点 q_idx 的 K 个最近邻"""
    start_time = time.time()
    W = []
    for i in range(1, N + 1):
        dist = distance_query_to_node(q_idx, i)
        # print(f"dist = {dist}")
        heapq.heappush(W, (-dist, i))
        if len(W) > K:
            heapq.heappop(W)
    ret = sorted([item[1] for item in W], reverse=True)
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    if print_flag:
        print("\n暴力搜索结果:", file=sys.stderr)
        print("序号：", ' '.join(map(str, ret)), file=sys.stderr)
        for r_idx in ret:
            dist = distance_query_to_node(q_idx, r_idx)
            print(f"{node[r_idx]} || {dist:.4f}", file=sys.stderr)
        print(f"查询耗时: {duration_ms:.4f} ms", file=sys.stderr)
    return ret

def genquerynode(num, dim=D):
    print("\n开始生成查询数据...", file=sys.stderr)
    global querynode
    for i in range(num):
        querynode.append([random.randint(MINX, MAXX) for _ in range(dim)])
    print("完成写入查询数据", file=sys.stderr)

def printgraph():
    tmp_edge = copy.deepcopy(edge)
    for x in tmp_edge:
        for i in range(MMAX0-len(x)):
            x.append(0)
    
    """把图的信息输出到 graph.txt"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "graph.txt")
    print("\n开始输出数据...", file=sys.stderr)

    with open(output_path, "w") as f:
        '''输出EnterPoint, L, 总点数 = len(edge) == len(ite)约等于N*M/(M-1)'''
        line = ' '.join([str(EP), str(L), str(len(tmp_edge))])
        f.write(line+'\n')
        print("完成写入EP 和 L", file = sys.stderr)

        '''输出edge'''
        for x in tmp_edge:
            line = ' '.join(str(_) for _ in x)
            f.write(line + '\n')
        print("完成写入edge", file = sys.stderr)

        '''输出ite'''
        line = ' '.join(str(_) for _ in ite)
        f.write(line+'\n')
        print("完成写入ite", file = sys.stderr)
def query(printflag = True):
    nq = 1000
    genquerynode(nq)
    global querynode
        
    print(f"\n主数据集大小 (node size): {len(node) - 1}", file=sys.stderr)
    print(f"查询集大小 (querynode size): {len(querynode)}", file=sys.stderr)

    if printflag:
        ave_recall = 0.0
        # --- 修改开始 (6/6): 修改查询循环和函数调用 ---
        for q_idx in range(nq):
            print("\n--------------------------------", file=sys.stderr)
            q_vec = querynode[q_idx]
            q_vec_str = ", ".join(map(str, q_vec))
            print(f"当前查询点 (索引 {q_idx}) 为 ({q_vec_str})", file=sys.stderr)
            
            print("\nq 的 K-ANN 搜索结果:", file=sys.stderr)
            start_query = time.time()
            # 调用查询专用函数 k_nn_search，并传递查询索引
            knn_res = k_nn_search(q_idx, K, EF_CONSTRUCTION)
            end_query = time.time()
            duration_query_ms = (end_query - start_query) * 1000

            print("序号: ",' '.join(map(str, knn_res)), file=sys.stderr)
            for res_idx in knn_res:
                dist = distance_query_to_node(q_idx, res_idx)
                print(f"{node[res_idx]} || {dist:.4f}", file=sys.stderr)
            print(f"查询耗时: {duration_query_ms:.4f} ms", file=sys.stderr)

            # 调用查询专用的暴力搜索函数
            brt_res = bruteforce_for_query(q_idx)
            
            brt_set = set(brt_res)
            ccnt = sum(1 for res_idx in knn_res if res_idx in brt_set)
            recall = ccnt / K
            ave_recall += recall
            print(f"召回率: {recall:.4f}", file=sys.stderr)
            print("-------------------------------------\n", file=sys.stderr)
    else:
        ave_recall = 0.0
        for q_idx in range(nq):
            # 调用查询专用函数 k_nn_search，并传递查询索引
            knn_res = k_nn_search(q_idx, K, EF_CONSTRUCTION)
            # 调用查询专用的暴力搜索函数
            brt_res = bruteforce_for_query(q_idx, False)
            
            brt_set = set(brt_res)
            ccnt = sum(1 for res_idx in knn_res if res_idx in brt_set)
            recall = ccnt / K
            ave_recall += recall

    # --- 修改结束 (6/6) ---
        
    print(f"\n平均召回率: {ave_recall / nq:.4f}", file=sys.stderr)
    

def main():
    """主执行函数"""
    gendata()
    # init()
    
    print("\n*********************", file=sys.stderr)
    print(f"N (数据点数): {N}", file=sys.stderr)
    print(f"D (维度): {D}", file=sys.stderr)
    print(f"M (最大连接数): {M}", file=sys.stderr)
    print(f"mL (层级因子): {ML:.4f}", file=sys.stderr)
    print(f"K (近邻数): {K}", file=sys.stderr)
    print(f"efConstruction: {EF_CONSTRUCTION}", file=sys.stderr)
    print("**********************\n", file=sys.stderr)

    start_insert = time.time()
    for i in range(1, N + 1):
        insert(i, M, MMAX, EF_CONSTRUCTION, ML)
        # if (i >= N) and (i % (N // 100) == 0):
        print(f"已插入节点数 = {i} / {N}", file=sys.stderr)
    end_insert = time.time()
    duration_insert = end_insert - start_insert
    print("\nHNSW 图构建完成", file=sys.stderr)
    print(f"建图耗时: {duration_insert:.4f} seconds", file=sys.stderr)
    
    # printgraph()
    # --- 修改开始 (5/6): 将查询点存入新的 querynode 列表 ---
    query(False)
    

if __name__ == "__main__":
    main()