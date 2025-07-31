HNSW文件夹里的文件是项目有用的文件\n
这个项目是./Programs/Source/ 这个文件夹下面的全部的文件\n
把HNSW整个文件夹放到./Programs/Source/内，或者直接放mp-spdz主目录下也可以。\n
在mp-spdz主目录下，bash 输入以下命令来编译和运行：\n
Scripts/compile-run.py ring HNSW/<文件名>.mpc\n
比如：\n
Scripts/compile-run.py ring HNSW/ball.mpc\n
输入输出文件：\n
Input-p1-0 存\n
hnsw_no_copied_node.txt 和 hnsw_copied_node.txt，分别为复制后的数据库，和原始未复制的数据库\n
Input-p2-0 存\n
hnsw_edge_graph.txt， 为建图后 相邻点信息\n