#include <bits/stdc++.h>
#include <chrono>
#include <random>
#include <unordered_set>
using namespace std;
#define foru(i, a, b) for (int i = a; i <= b; ++i)
#define ford(i, a, b) for (int i = a; i >= b; --i)
#define pii pair<int, int>
#define prt(x) cerr << #x << " : " << x << endl;
// #define int long long

const int minx = -1000;
const int maxx = 1000;

const int d = 4;
const int M = 16;
const int Mmax0 = 2 * M;
const int Mmax = M;
const int n = 1e4;
const int efConstruction = 10;
const int K = 1;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> uni(0.0, 1.0);

int L, EP = 1, nodecnt = 0;
double mL = 1 / log(M);

vector<vector<int>> edge;
vector<vector<int>> node(n + 1, vector<int>(d));
vector<int> ite;
// vector<int> tp(n + 1);

void gendata(int num = n, int dim = d) {
    cerr << "Start gendata\n";
    ofstream fout("data.txt");
    std::uniform_int_distribution<int> uniform_dist(minx, maxx);
    foru(i, 1, num) {
        foru(j, 1, dim) {
            fout << uniform_dist(gen) << " \n"[j == dim];
        }
    }
    fout.close();
    cerr << "done writing data.txt\n";
}

int distance(int q, int x) {
    vector<int>&nq = node[q], &nx = node[x];
    long long res = 0;
    foru(i, 0, d - 1) res += (long long)(nq[i] - nx[i]) * (long long)(nq[i] - nx[i]);
    return (int)(sqrt(res));
}

vector<int> NXT(vector<int> W) {
    foru(i, 0, (int)(W.size()) - 1) W[i]++;
    return W;
}

vector<int> SEARCH_LAYER(int q, vector<int> ep, int ef, int lc) {
    unordered_set<int> v;
    for (auto x : ep)
        v.insert(x);
    priority_queue<pii, vector<pii>, greater<pii>> C;
    for (auto x : ep)
        C.push(make_pair(distance(q, ite[x]), x));
    priority_queue<pair<int, int>> W;
    for (auto x : ep)
        W.push(make_pair(distance(q, ite[x]), x));

    while (!C.empty()) {
        auto cc = C.top();
        C.pop();
        int c = cc.second, discq = cc.first;
        auto ff = W.top();
        int f = ff.second, disfq = ff.first;
        if (discq > disfq)
            break;
        for (auto e : edge[c])
            if (!v.count(e)) {
                v.insert(e);
                int diseq = distance(q, ite[e]);
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
    return ret;
}

vector<int> SELECT_NEIGHBORS_SIMPLE(int q, vector<int> C, int M, int lc) {
    priority_queue<pair<int, int>> W;
    for (auto x : C) {
        W.push(make_pair(distance(q, ite[x]), x));
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
vector<int> SELECT_NEIGHBORS_HEURISTIC(int q, vector<int> C, int M, int lc, bool extandCandidates = false, bool keepPrunedConnections = true) {
    priority_queue<pii, vector<pii>, greater<pii>> R;
    priority_queue<pii, vector<pii>, greater<pii>> W;
    for (auto x : C)
        W.push(make_pair(distance(q, ite[x]), x));

    if (extandCandidates) {
        unordered_set<int> st;
        for (auto x : C)
            st.insert(x);
        for (auto e : C) {
            for (auto eadj : edge[e])
                if (!st.count(eadj)) {
                    st.insert(eadj);
                    W.push(make_pair(distance(q, ite[eadj]), eadj));
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

int rres = 0;
int rcnt[10];
double rans = 0;
void INSERT(int q, int M, int Mmax, int efConstruction, double mL) {
    static std::uniform_real_distribution<double> uni(0.0, 1.0);
    vector<int> ep;
    if (nodecnt != 0)
        ep.emplace_back(EP);

    int l = floor(-log(uni(gen)) * mL);
    if (l > L) {
        EP = nodecnt + 1;  // q at layer lc
        ford(lc, l, L + 1) {
            ++nodecnt;
            ite.emplace_back(q);
            edge.emplace_back(vector<int>());
        }
    }
    // tp[q] = nodecnt + 1;
    rans += (l + 1);
    // cerr << "l : " << l << endl;
    rcnt[l]++;
    rres = max(rres, l);

    vector<int> W;
    ford(lc, L, l + 1) {
        W = SEARCH_LAYER(q, ep, 1, lc);
        assert((int)(W.size()) == 1);
        ep = NXT(W);
    }
    vector<int> neighbors, eConn, eNewConn;

    ford(lc, min(L, l), 0) {
        if (lc == 0)
            Mmax = Mmax0;
        ++nodecnt;  // q at layer lc
        ite.emplace_back(q);
        W = SEARCH_LAYER(q, ep, efConstruction, lc);
        assert((int)(W.size()) <= efConstruction);
        neighbors = SELECT_NEIGHBORS_HEURISTIC(q, W, M, lc);
        edge.emplace_back(neighbors);  // edge[q at layer lc]
        for (auto e : neighbors) {
            edge[e].emplace_back(nodecnt);
            eConn = edge[e];
            if ((int)(eConn.size()) > Mmax) {
                eNewConn = SELECT_NEIGHBORS_HEURISTIC(ite[e], eConn, Mmax, lc);
                edge[e] = eNewConn;
            }
        }
        ep = NXT(W);
    }
    if (l > L)
        L = l;
}

vector<int> K_NN_SEARCH(int q, int K, int ef) {
    vector<int> W;
    vector<int> ep;
    ep.emplace_back(EP);
    ford(lc, L, 1) {
        W = SEARCH_LAYER(q, ep, 1, lc);
        assert((int)(W.size()) == 1);
        ep = NXT(W);
    }
    W = SEARCH_LAYER(q, ep, ef, 0);
    priority_queue<pii> W_;
    for (auto x : W) {
        W_.push(make_pair(distance(q, ite[x]), ite[x]));
        if ((int)(W_.size()) > K)
            W_.pop();
    }
    vector<int> ret;
    while (!W_.empty()) {
        ret.emplace_back(W_.top().second);
        W_.pop();
    }
    return ret;
}

void init() {
    cerr << "Start reading data\n";
    ifstream fin("data.txt");
    int i = 1, x, j = 0;
    while (fin >> x) {
        node[i][j] = x;
        if (j == d - 1) {
            ++i;
            j = 0;
        } else
            ++j;
    }
    fin.close();
    cerr << "done reading data.txt\n";
    ite.push_back(0);
    edge.push_back(vector<int>());
    foru(ii, 1, 3)
            foru(i, 0, d - 1) cerr
        << node[ii][i] << " \n"[i == d - 1];
}
vector<int> bruteforce(int q) {
    auto start = std::chrono::high_resolution_clock::now();
    priority_queue<pii> W;

    foru(i, 1, n) {
        W.push(make_pair(distance(q, i), i));
        if ((int)(W.size()) > K)
            W.pop();
    }
    vector<int> ret;
    while (!W.empty()) {
        ret.push_back(W.top().second);
        W.pop();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    cerr << "\nBruteforce :\n";
    foru(i, 0, K - 1) cerr << ret[i] << " \n"[i == K - 1];
    foru(i, 0, K - 1) {
        foru(j, 0, d - 1) cerr << node[ret[i]][j] << " ";
        cerr << "|| " << distance(ret[i], q) << endl;
    }
    cerr << "query_runtime : " << duration.count() << " ms" << "\n";
    return ret;
}

signed main() {
    gendata();
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

    auto start = std::chrono::high_resolution_clock::now();
    foru(i, 1, n) {
        INSERT(i, M, Mmax, efConstruction, mL);
        if (i % (n / 100) == 0)
            cerr << "i = " << i << "\n";
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_insert = end - start;
    cerr << "\ndone building the HNSW\n";

    prt(rres);
    foru(i, 0, rres) prt(rcnt[i]);
    rans = rans / n;
    prt(rans);

    int nq = 21;
    cerr << "Input # of queries : ";  // cin >> nq;
    foru(i, n + 1, n + nq) {
        node.emplace_back(vector<int>(d, 0));
        foru(j, 0, d - 1) node[i][j] = (i - n - (nq + 1) / 2) * ((maxx - minx) / (nq - 1));
    }
    cerr << "\nnode.size : " << (int)(node.size()) << endl;

    double ave_recall = 0;
    foru(q, n + 1, n + nq) {
        // cerr << "Input query vector(of dimension " << d << ", format**** ... **):\n";
        // node.push_back(vector<int>(d));
        /*foru(j, 0, d - 1) {
                cin >> node[i][j];
        }*/
        cerr << "\n--------------------------------\nnow q is (";
        foru(j, 0, d - 1) {
            cerr << node[q][j] << ",)"[j == d - 1];
            if (j == d - 1)
                cerr << "\n";
        }

        cerr << "\nK-ANN of q: \n";
        auto start = std::chrono::high_resolution_clock::now();
        vector<int> knn_res = K_NN_SEARCH(q, K, efConstruction);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        foru(j, 0, K - 1) cerr << knn_res[j] << " \n"[j == K - 1];
        foru(i, 0, K - 1) {
            foru(j, 0, d - 1) cerr << node[knn_res[i]][j] << " ";
            cerr << "|| " << distance(knn_res[i], q) << endl;
        }
        cerr << "query_runtime : " << duration.count() << " ms" << "\n";

        auto brt_res = bruteforce(q);
        cerr << "\n recall : ";
        unordered_set<int> st;
        foru(i, 0, K - 1) st.insert(brt_res[i]);
        int ccnt = 0;
        foru(i, 0, K - 1) if (st.count(knn_res[i])) ccnt++;
        vector<double> recalls;
        recalls.emplace_back((double)(ccnt) / (double)(K));
        cerr << recalls[(int)(recalls.size()) - 1] << "\n";
        ave_recall += recalls[(int)(recalls.size()) - 1];
        cerr << "-------------------------------------\n";
    }
    cerr << "Average Recall :" << ave_recall / nq << "\n";
    cerr << "Insert_runtime : " << duration_insert.count() << " seconds" << "\n";
    return 0;
}