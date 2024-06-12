#include "Node.h"
#include "roll.cuh"
#include <ctime>
#include <iostream>
#include <thread>
#include <valarray>
#include <mutex>
#define COMPLETE 3
#define PARTLY 4
#define ALL_NOT 5
#define ROLLTIMES 1024
void roll_and_update_GPU(Node* root, int roll_times);
void roll_GPUthread(Node* child, const int roll_times);
using namespace std;
recursive_mutex mut;

Node::Node(State *state, const bool copy=false) {
    if (copy) this->state = new State(state->chess);
    else this->state = state;

    // 初始化孩子
    this->child_nums = int(this->state->workable_place.size());
    this->child_nums_expanded = 0;
    /*for (int i = 0; i < child_nums; i++) {
        //this->none_child.push_back(this->state->workable_place[i]);
        // this->state->workable_place.pop_back();
    }*/
}

Node::~Node() {
    delete this->state;
    this->state = nullptr;
    this->parent = nullptr;
    while (!this->child.empty()) this->child.pop_back();
}

float Node::UCB(const float C=2) {
    if(this->parent != nullptr) {
        this->parent->UCB();
        this->score = this->win_times / this->visit + C * std::sqrt(std::log(this->parent->visit) / this->visit);
    }
    return this->score;
}


void Node::add_child(State *child_state, bool copy) {
    const auto node = new Node(child_state, copy);
    node->parent = this;
    node->state->chess->judge();
    node->state->leaf = node->state->chess->winner;
    this->child.push_back(node);

}

int Node::full_expand() const {
    // 有三种 子结点展开状态,分别返回：未展开，所有都展开，完全展开
    if (this->child_nums_expanded == 0) return ALL_NOT;
    if (child_nums_expanded == child_nums) {
        return COMPLETE;
    };
    return PARTLY;
}

Node* Node::choice() const {
    // 从当前结点中选择一个最合适的子节点：未完全展开 中的 UCB最大
    Node* best_child = nullptr;
    // 如果本次为player_1搜索，越大越好
    if(this->state->chess->who_place_next == 1){
        float biggest_score = 0;
        for (const auto i : this->child) {
            if (i->full_expand() != COMPLETE && i->score > biggest_score) {
                best_child = i;
                biggest_score = i->score;
            }
        }
    }
    // 如果为player_0搜索，越小越好
    else {
        float smallest_score = 1e9;
        for (const auto i : this->child) {
            if (i->full_expand() != COMPLETE && i->score < smallest_score) {
                best_child = i;
                smallest_score = i->score;
            }
        }
    }
    //if (best_child == nullptr) {
        //this->state->chess->show_mat();
    //}  // 注意有可能已经收敛，所有的都已经完全展开
    return best_child;

}

int Node::expand() {
    // 拓展函数，作用：将该节点展开，构造出所有子结点
    // 找到所有的可行位，落子然后构造一个子节点，加入数组
    for (const int i : this->state->workable_place) {
        const int y_ = i % this->state->chess->length;
        const int x_ = (i - y_) / this->state->chess->length;
        this->add_child(new State(this->state->chess, x_, y_), true);
        if (this->child.back()->state->leaf != -1) {
            this->child.back()->expended = 1;
        }
    }
    this->expended = 1;  // 当前结点已展开
    if (this->parent != nullptr) this->parent->child_nums_expanded++;  // 当前结点对应的父节点的已展开结点数+1
    return 1;
}

void Node::update(const float visit_add, float win_times) {
    // 更新当前结点
    this->visit = this->visit + visit_add;
    this->win_times = this->win_times + win_times;
    // 递归更新父结点
    if (this->parent != nullptr) {
        this->parent->update(visit_add, win_times);
    }
    // 更新UCB的值-----更正：由于遍历顺序原因，UCB更新应当在遍历完子节点之后
    //this->UCB(2);
}


std::vector<int>* find_place(const Node *node) {
    // 找到某一个结点上可以落子的所有位置
    auto *place = new std::vector<int>;
    for (int i = 0; i < node->state->chess->length*node->state->chess->length; i++) {
        if (node->state->chess->mat[i] == -1) place->push_back(i);
    }
    return place;
}

void rollout(Node *node, const std::vector<int> *place, int roll_times=1024) {
    // 对某一个结点进行rollout，在一定次数以内随机落子
    float  times=0, win_times=0;
    if (node->state->leaf == 1) {
        mut.lock();
        // 对模拟结点进行反向传播
        node->update(1.0, 1.0);
        mut.unlock();
        return;
    }
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    while(roll_times > 0) {
        times++;
        roll_times--;

        // 每次棋局 创建结点与落子坐标的副本
        std::vector<int> temp_vector = *place;
        const auto* temp = new Node(node->state, true);
        while(temp->state->chess->winner == -1 && !temp_vector.empty()) {
            constexpr int min_value = 0;
            const int max_value = int(temp_vector.size())-1;

            // 计算vector内的随机index
            const int index = min_value + std::rand() % (max_value - min_value + 1);

            // 落子
            const int y = (temp_vector[index] % temp->state->chess->length);
            const int x = (temp_vector[index] - y) / temp->state->chess->length;
            temp->state->chess->put(x, y);
            // 丢弃走过的位置
            temp_vector.erase(temp_vector.begin() + index);
            // 判断
            temp->state->chess->judge();
        }
        if (temp->state->chess->winner != -1) win_times = win_times + static_cast<float>(temp->state->chess->winner);
        delete temp;
    }
    // 对模拟结点进行反向传播
    mut.lock();
    node->update(1, win_times/times);
    mut.unlock();
    delete place;
}

int best_answer(const Node *node) {
    int answer = -1;
    float max_score = 0;
    for (int i = 0; i < node->child.size(); i++) {
        if (node->child[i]->score > max_score) {
            answer = i;
            max_score = node->child[i]->score;
        }
    }
    return node->child[answer]->state->chess->place_index_last;
}


int MCTS_search(Chess* root_chess, const float time_limit) {
    // 初始化时间变量和答案变量
    const clock_t start = clock();
    clock_t end = clock();
    int answer = -1;

    // 初始化状态根结点，并初次拓展与模拟，设置一个用于查找的temp指针
    auto root = Node(new State(root_chess));
    root.expand();
    roll_and_update_GPU(&root, ROLLTIMES);
    //for(const auto & i : root.child) {
    //    rollout(i, find_place(i), 104);
    //}
    // for(const auto & i : root.child) {
    //     i->UCB();
    // }
    Node *temp = &root;
    // 在时间范围内进行迭代优化,每次从根开始查找
    while (end - start <= time_limit) {
        temp = &root;
        temp = temp->choice();
        if (temp == nullptr) {
            // 所有子结点都已经完全展开，收敛
            break;
        }
        // 存在未完全展开的字结点，不断展开到未曾展开的子节点
        while (temp->expended) {
            auto *_ = temp->choice();
            if (_ == nullptr) {
                goto end;  // 在搜索时达到一个胜负点，说明此时已经收敛
            }
            temp = _;
        }
        temp->expand();
        roll_and_update_GPU(temp, ROLLTIMES);
        //roll_and_update(temp, ROLLTIMES);

        // 废弃的单线程实现
        // for(const auto & i : temp->child) {
        //     rollout(i, find_place(i), 1000);
        // }
        // for(const auto & i : temp->child) {
        //     i->UCB();
        // }
        end = clock();
    }

    end:
    answer = best_answer(&root);
    return answer;
}

void roll_thread(Node *child,const int roll_times) {
    rollout(child, find_place(child), roll_times);
}

void UCB_thread(Node *child) {
    child->UCB();
}

void roll_and_update(Node* root,int roll_times) {
    std::vector<std::thread> pool;
    for(int i = 0; i < root->child.size(); i++) {
        pool.push_back(std::thread(roll_thread, root->child[i], 1024));
    }
    for(int i = 0; i < root->child.size(); i++) {
        pool[i].join();
    }

    for(int i = 0; i < root->child.size(); i++) {
        pool[i] = std::thread(UCB_thread, root->child[i]);
    }
    for(int i = 0; i < root->child.size(); i++) {
        pool[i].join();
    }
};

// 以下为GPU版本的函数
void roll_GPUthread(Node* child, const int roll_times) {
    std::vector<int>* place = find_place(child);
    int place_eles = int(place->size());
    int* place_arr = new int[place->size()];
    std::copy(place->begin(), place->end(), place_arr);
    delete place;

    rollout_GPU(child, child->state->chess->length, place_arr, place_eles, roll_times);
}


void roll_and_update_GPU(Node* root, int roll_times) {
    std::vector<std::thread> pool;
    for (int i = 0; i < root->child.size(); i++) {
        pool.push_back(std::thread(roll_GPUthread, root->child[i], roll_times));
    }
    for (int i = 0; i < root->child.size(); i++) {
        pool[i].join();
    }

    for (int i = 0; i < root->child.size(); i++) {
        pool[i] = std::thread(UCB_thread, root->child[i]);
    }
    for (int i = 0; i < root->child.size(); i++) {
        pool[i].join();
    }
};