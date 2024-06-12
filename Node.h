﻿#ifndef NODE_H
#define NODE_H
#include "State.h"

class Node {
public:
    State *state = nullptr;  // 结点对应状态
    float score=3.402823466e+38F, visit=0, win_times=0;  // 结点对应的分数，访问次数，赢下次数
    int child_nums_expanded=0, child_nums=0, expended=0;  // 展开的孩子数量，孩子数量，自己是否展开过
    //int leaf=-1;  // 是否叶节点？(转移到了State上面)
    //std::vector<int> none_child{};  // 所有尚未生成的孩子结点，存储孩子的可落子位置(可以用state的workable，似乎用不上)
    std::vector<Node*> child{};  // 展开过的孩子结点
    Node *parent=nullptr;  // 父母结点
    Node(State* state, bool copy);
    ~Node();
    float UCB(float C);
    void add_child(State* child_state, bool copy);  // 添加孩子结点
    int full_expand() const;  // 判断所有子结点都展开函数
    Node* choice() const;  // 选择函数
    int expand();  //  拓展函数
    // 模拟函数
    void update(float visit_add, float win_times);  // 反向传播
};

// 辅助函数
std::vector<int>* find_place(const Node* node);  // 寻路函数，找到空位置
void rollout(Node* node, const std::vector<int>& place, float roll_times);  // 模拟函数(串行版本)
int best_answer(const Node* node);
int MCTS_search(Chess* root_chess,float time_limit);
// 多线程部分
void roll_thread(Node* child,int roll_times);
void UCB_thread(Node *child);
void roll_and_update(Node* child,int roll_times);
// GPU加速部分
void roll_GPUthread(Node* child, const int roll_times);
void roll_and_update_GPU(Node* root, int roll_times);
#endif //NODE_H
