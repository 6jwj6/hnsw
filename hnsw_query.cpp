#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

using namespace std;
#define foru(i, a, b) for (int i = a; i <= b; ++i)
#define ford(i, a, b) for (int i = a; i >= b; --i)
#define pii pair<int, int>
#define prt(x) cerr << #x << " : " << x << endl;

// --- 全局常量和变量 ---
const int minx = -100;
const int maxx = 100;

const int d = 4;
const int M = 6;
const int Mmax0 = 2 * M;
const int Mmax = M;
const int n = 10000;
const int efConstruction = 10;
const int K = 1;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> uni(0.0, 1.0);

int L, EP = 1, nodecnt = 0;
double mL = 1 / log(M);

// --- 修改: 数据容器分离 ---
vector<vector<int>> node;
vector<vector<int>> querynode;
// --- 修改结束 ---

vector<vector<int>> edge;
vector<int> ite;

void gendata(int num = n, int dim = d) {
    cerr << "Start gendata\n";
    ofstream fout("data.txt");
    std::uniform_int_distribution<int> uniform_dist(minx, maxx);
    foru(i, 1, num) {
        foru(j, 1, dim) {
            fout << uniform_dist(gen) << (j == dim ? "" : " ");
        }
        fout << "\n";
    }
    fout.close();
    cerr << "done writing data.txt\n";
}

void genquerynode(int num, int dim = d){
    std::uniform_int_distribution<int> uniform_dist(minx, maxx);
    foru(i, 0, num-1) {
        querynode.emplace_back(vector<int>(dim, 0));
        foru(j, 0, dim-1) {
            querynode[i][j] = uniform_dist(gen);
        }
    }
}
// --- 修改: 创建两个版本的 distance 函数，逻辑与原始版本完全相同 ---
int distance_node_to_node(int idx1, int idx2) {
    vector<int>& nq = node[idx1];
    vector<int>& nx = node[idx2];
    long long res = 0;
    foru(i, 0, d - 1) res += (long long)(nq[i] - nx[i]) * (long long)(nq[i] - nx[i]);
    return (int)(sqrt(res));
}

int distance_query_to_node(int q_idx, int node_idx) {
    vector<int>& nq = querynode[q_idx];
    vector<int>& nx = node[node_idx];
    long long res = 0;
    foru(i, 0, d - 1) res += (long long)(nq[i] - nx[i]) * (long long)(nq[i] - nx[i]);
    return (int)(sqrt(res));
}
// --- 修改结束 ---

vector<int> NXT(vector<int> W) {
    // 原始代码中此函数有误，应传递引用或重新赋值。此处修正以符合预期行为。
    for (int& val : W)
        val++;
    return W;
}

// --- HNSW 核心算法 ---

// --- 修改: SEARCH_LAYER 专用于插入流程, 内部调用 distance_node_to_node ---
vector<int> SEARCH_LAYER(int q, vector<int> ep, int ef, int lc) {
    unordered_set<int> v;
    for (auto x : ep)
        v.insert(x);
    priority_queue<pii, vector<pii>, greater<pii>> C;
    for (auto x : ep)
        C.push(make_pair(distance_node_to_node(q, ite[x]), x));
    priority_queue<pii> W;
    for (auto x : ep)
        W.push(make_pair(distance_node_to_node(q, ite[x]), x));

    while (!C.empty()) {
        auto cc = C.top();
        C.pop();
        int c = cc.second, discq = cc.first;
        // 严格遵循原始代码的逻辑
        auto ff = W.top();
        int f = ff.second, disfq = ff.first;
        if (discq > disfq)
            break;
        for (auto e : edge[c])
            if (!v.count(e)) {
                v.insert(e);
                int diseq = distance_node_to_node(q, ite[e]);
                // 严格遵循原始代码的逻辑
                disfq = W.top().first;
                if (diseq < disfq || (int)(W.size()) < ef) {
                    C.push(make_pair(diseq, e));
                    W.push(make_pair(diseq, e));
                    if ((int)(W.size()) > ef)
                        W.pop();
                }
            }
    }
    vector<int> ret;
    while (!W.empty()) {
        ret.emplace_back(W.top().second);
        W.pop();
    }
    reverse(ret.begin(), ret.end());
    return ret;
}
// --- 修改结束 ---

// --- 修改: 为查询流程创建 search_layer_for_query, 逻辑与原始 SEARCH_LAYER 完全一致 ---
vector<int> SEARCH_LAYER_FOR_QUERY(int q, vector<int> ep, int ef, int lc) {
    // if (lc > 0) return ep;
    unordered_set<int> v;
    for (auto x : ep)
        v.insert(x);
    priority_queue<pii, vector<pii>, greater<pii>> C;
    for (auto x : ep)
        C.push(make_pair(distance_query_to_node(q, ite[x]), x));
    priority_queue<pii> W;
    for (auto x : ep)
        W.push(make_pair(distance_query_to_node(q, ite[x]), x));

    while (!C.empty()) {
        auto cc = C.top();
        C.pop();
        int c = cc.second, discq = cc.first;
        // 严格遵循原始代码的逻辑
        auto ff = W.top();
        int f = ff.second, disfq = ff.first;
        if (discq > disfq)
            break;
        for (auto e : edge[c])
            if (!v.count(e)) {
                v.insert(e);
                int diseq = distance_query_to_node(q, ite[e]);
                // 严格遵循原始代码的逻辑
                disfq = W.top().first;
                if (diseq < disfq || (int)(W.size()) < ef) {
                    C.push(make_pair(diseq, e));
                    W.push(make_pair(diseq, e));
                    if ((int)(W.size()) > ef)
                        W.pop();
                }
            }
    }
    vector<int> ret;
    while (!W.empty()) {
        ret.emplace_back(W.top().second);
        W.pop();
    }
    reverse(ret.begin(), ret.end());
    return ret;
}
// --- 修改结束 ---
vector<int> SELECT_NEIGHBORS_SIMPLE(int q, vector<int> C, int M, int lc) {
    priority_queue<pair<int, int>> W;
    for (auto x : C) {
        W.push(make_pair(distance_node_to_node(q, ite[x]), x));
        if ((int)(W.size()) > M)
            W.pop();
    }
    vector<int> ret;
    while (!W.empty()) {
        ret.emplace_back(W.top().second);
        W.pop();
    }
    return ret;
}
// --- 修改: SELECT_NEIGHBORS_HEURISTIC 内部调用专用的距离函数 ---
vector<int> SELECT_NEIGHBORS_HEURISTIC(int q, vector<int> C, int M, int lc, bool extandCandidates = false, bool keepPrunedConnections = true) {
    // 要改掉，现在是错的
    priority_queue<pii, vector<pii>, greater<pii>> R;
    priority_queue<pii, vector<pii>, greater<pii>> W;
    for (auto x : C)
        W.push(make_pair(distance_node_to_node(q, ite[x]), x));

    if (extandCandidates) {
        unordered_set<int> st;
        for (auto x : C)
            st.insert(x);
        for (auto e : C) {
            for (auto eadj : edge[e])
                if (!st.count(eadj)) {
                    st.insert(eadj);
                    W.push(make_pair(distance_node_to_node(q, ite[eadj]), eadj));
                }
        }
    }
    priority_queue<pii, vector<pii>, greater<pii>> Wd;
    while (!W.empty() && (int)(R.size()) < M) {
        auto ee = W.top();
        W.pop();
        int e = ee.second, diseq = ee.first;
        if (R.empty() || diseq < R.top().first)
            R.push(ee);
        else
            Wd.push(ee);
    }
    if (keepPrunedConnections) {
        while (!Wd.empty() && (int)(R.size()) < M) {
            R.push(Wd.top());
            Wd.pop();
        }
    }
    vector<int> ret;
    while (!R.empty()) {
        ret.emplace_back(R.top().second);
        R.pop();
    }
    return ret;
}
// --- 修改结束 ---

// INSERT 函数逻辑完全不变，它会正确调用专用于插入的 SEARCH_LAYER
void INSERT(int q, int M_val, int Mmax_val, int efConstruction_val, double mL_val) {
    vector<int> ep;
    if (nodecnt != 0)
        ep.emplace_back(EP);
    int l = floor(-log(uni(gen)) * mL_val);
    if (l > L) {
        EP = nodecnt + 1;
        ford(lc, l, L + 1) {
            ++nodecnt;
            ite.emplace_back(q);
            edge.emplace_back(vector<int>());
        }
    }
    vector<int> W;
    ford(lc, L, l + 1) {
        W = SEARCH_LAYER(q, ep, 1, lc);
        assert((int)(W.size()) == 1);
        ep = NXT(W);
    }
    ford(lc, min(L, l), 0) {
        int current_Mmax = (lc == 0) ? Mmax0 : Mmax_val;
        ++nodecnt;
        ite.emplace_back(q);
        W = SEARCH_LAYER(q, ep, efConstruction_val, lc);
        vector<int> neighbors = SELECT_NEIGHBORS_SIMPLE(q, W, M_val, lc);
        edge.emplace_back(neighbors);
        for (auto e : neighbors) {
            edge[e].emplace_back(nodecnt);
            if ((int)(edge[e].size()) > current_Mmax) {
                vector<int> eNewConn = SELECT_NEIGHBORS_SIMPLE(ite[e], edge[e], current_Mmax, lc);
                edge[e] = eNewConn;
            }
        }
        ep = NXT(W);
    }
    if (l > L)
        L = l;
}

// --- 修改: K_NN_SEARCH 调用查询专用函数 ---
vector<int> K_NN_SEARCH(int q, int K_val, int ef_val) {
    vector<int> W;
    vector<int> ep;
    ep.emplace_back(EP);
    ford(lc, L, 1) {
        W = SEARCH_LAYER_FOR_QUERY(q, ep, 1, lc);
        assert((int)(W.size()) == 1);
        ep = NXT(W);
        // ep = NXT(ep);
    }
    W = SEARCH_LAYER_FOR_QUERY(q, ep, ef_val, 0);

    priority_queue<pii> W_;
    for (auto x : W) {
        W_.push(make_pair(distance_query_to_node(q, ite[x]), ite[x]));
        if ((int)(W_.size()) > K_val)
            W_.pop();
    }
    vector<int> ret;
    while (!W_.empty()) {
        ret.emplace_back(W_.top().second);
        W_.pop();
    }
    reverse(ret.begin(), ret.end());
    return ret;
}
// --- 修改结束 ---

// --- 修改: bruteforce 创建查询专用版本 ---
vector<int> bruteforce_for_query(int q) {
    auto start = std::chrono::high_resolution_clock::now();
    priority_queue<pii> W;

    foru(i, 1, n) {
        W.push(make_pair(distance_query_to_node(q, i), i));
        if ((int)(W.size()) > K)
            W.pop();
    }
    vector<int> ret;
    while (!W.empty()) {
        ret.push_back(W.top().second);
        W.pop();
    }
    reverse(ret.begin(), ret.end());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    cerr << "\nBruteforce :\n";
    for (size_t i = 0; i < ret.size(); ++i)
        cerr << ret[i] << (i == ret.size() - 1 ? "" : " ");
    cerr << endl;
    for (int res_idx : ret) {
        foru(j, 0, d - 1) cerr << node[res_idx][j] << " ";
        cerr << "|| " << distance_query_to_node(q, res_idx) << endl;
    }
    cerr << "query_runtime : " << duration.count() << " ms" << "\n";
    return ret;
}
// --- 修改结束 ---

void init() {
    cerr << "Start reading data\n";
    ifstream fin("data.txt");
    node.assign(n + 1, vector<int>(d));
    foru(i, 1, n) {
        foru(j, 0, d - 1) {
            fin >> node[i][j];
        }
    }
    fin.close();
    cerr << "done reading data.txt\n";
    ite.push_back(0);
    edge.push_back(vector<int>());
    foru(ii, 1, 3) {
        if (ii > n)
            break;
        foru(i, 0, d - 1) cerr << node[ii][i] << (i == d - 1 ? "" : " ");
        cerr << endl;
    }
}

signed main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    // gendata();
    init();

    cerr << "\n*********************" << endl;
    prt(n);
    prt(d);
    prt(M);
    prt(mL);
    prt(K);
    prt(efConstruction);
    cerr << "**********************\n"
         << endl;

    auto start_build = std::chrono::high_resolution_clock::now();
    foru(i, 1, n) {
        INSERT(i, M, Mmax, efConstruction, mL);
        if (i > 0 && i % (n / 100) == 0)
            cerr << "i = " << i << "\n";
    }
    auto end_build = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_insert = end_build - start_build;
    cerr << "\ndone building the HNSW\n";

    // 省略了原始代码中一些统计打印

    // --- 修改: 查询点存入 querynode, 并修改调用方式 ---
    int nq = 1000;
    cerr << "Number of queries: " << nq << endl;
    genquerynode(nq);
    // querynode.assign(nq, vector<int>(d));
    // foru(i, 0, nq - 1) {
    //     double val = (i - (double)(nq - 1) / 2.0) * ((double)(maxx - minx) / (double)(nq - 1));
    //     foru(j, 0, d - 1) querynode[i][j] = static_cast<int>(val);
    // }
    cerr << "\nnode.size : " << n << endl;
    cerr << "querynode.size : " << querynode.size() << endl;

    double ave_recall = 0;
    foru(i, 0, nq - 1) {
        int q_idx = i;

        cerr << "\n--------------------------------\nnow q is (";
        foru(j, 0, d - 1) cerr << querynode[q_idx][j] << (j == d - 1 ? ",)" : ", ");
        cerr << "\n";

        cerr << "\nK-ANN of q: \n";
        auto start_query = std::chrono::high_resolution_clock::now();
        vector<int> knn_res = K_NN_SEARCH(q_idx, K, efConstruction);
        auto end_query = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_query = end_query - start_query;

        for (size_t j = 0; j < knn_res.size(); ++j)
            cerr << knn_res[j] << (j == knn_res.size() - 1 ? "" : " ");
        cerr << endl;
        for (int res_idx : knn_res) {
            foru(j, 0, d - 1) cerr << node[res_idx][j] << " ";
            cerr << "|| " << distance_query_to_node(q_idx, res_idx) << endl;
        }
        cerr << "query_runtime : " << duration_query.count() << " ms" << "\n";

        auto brt_res = bruteforce_for_query(q_idx);

        cerr << "\nrecall : ";
        unordered_set<int> st;
        for (int res_idx : brt_res)
            st.insert(res_idx);
        int ccnt = 0;
        for (int res_idx : knn_res)
            if (st.count(res_idx))
                ccnt++;

        double recall = (double)(ccnt) / (double)(K);
        cerr << recall << "\n";
        ave_recall += recall;
        cerr << "-------------------------------------\n";
    }
    cerr << "Average Recall :" << ave_recall / nq << "\n";
    cerr << "Insert_runtime : " << duration_insert.count() << " seconds" << "\n";

    return 0;
}
// --- 修改结束 ---