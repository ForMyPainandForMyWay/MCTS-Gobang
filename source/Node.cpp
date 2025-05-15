#include "Node.h"
#include "roll.cuh"
#include <ctime>
#include <thread>
#include <cmath>
#include <mutex>
#define COMPLETE 3
#define PARTLY 4
#define ALL_NOT 5
#define ROLL_TIMES 1024


std::recursive_mutex mut;

Node::Node(State *state) {
    this->state = state;

    // 初始化孩子
    this->ChildNums = static_cast<int>(this->state->WorkablePlace.size());
    this->ChildNumsExpanded = 0;
}

Node::~Node() {
    delete this->state;
    this->state = nullptr;
    this->parent = nullptr;
    for (const auto child : this->Child)delete child;
    this->Child.clear();
}

float Node::UCB(const float C=2) {
    if(this->parent != nullptr) {
        this->parent->UCB();
        this->score = this->win_times / this->visit + C * std::sqrt(std::log(this->parent->visit) / this->visit);
    }
    return this->score;
}


void Node::AddChild(State *state) {
    const auto node = new Node(state);
    node->parent = this;
    node->state->chess->Judge();
    node->state->Lear = node->state->chess->Winner;
    this->Child.push_back(node);
}

int Node::FullExpand() const {
    // 有三种 子结点展开状态,分别返回：未展开，所有都展开，完全展开
    if (this->ChildNumsExpanded == 0) return ALL_NOT;
    if (ChildNumsExpanded == ChildNums) {
        return COMPLETE;
    };
    return PARTLY;
}

Node* Node::Choice() const {
    // 从当前结点中选择一个最合适的子节点：未完全展开 中的 UCB最大
    Node* best_child = nullptr;
    // 如果本次为player_1搜索，越大越好
    if(this->state->chess->WhoPlaceNext == 1){
        float biggest_score = 0;
        for (const auto i : this->Child) {
            if (i->FullExpand() != COMPLETE && i->score > biggest_score) {
                best_child = i;
                biggest_score = i->score;
            }
        }
    }
    // 如果为player_0搜索，越小越好
    else {
        float smallest_score = 1e9;
        for (const auto i : this->Child) {
            if (i->FullExpand() != COMPLETE && i->score < smallest_score) {
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

int Node::Expand() {
    // 拓展函数，作用：将该节点展开，构造出所有子结点
    // 找到所有的可行位，落子然后构造一个子节点，加入数组
    for (const int i : this->state->WorkablePlace) {
        const int y_ = i % this->state->chess->Length;
        const int x_ = (i - y_) / this->state->chess->Length;
        this->AddChild(new State(this->state->chess, x_, y_));
        if (this->Child.back()->state->Lear != -1) {
            this->Child.back()->Expended = 1;
        }
    }
    this->Expended = 1;  // 当前结点已展开
    if (this->parent != nullptr) this->parent->ChildNumsExpanded++;  // 当前结点对应的父节点的已展开结点数+1
    return 1;
}

void Node::Update(const float visit_add, const float win_times) {
    // 更新当前结点
    this->visit = this->visit + visit_add;
    this->win_times = this->win_times + win_times;
    // 递归更新父结点
    if (this->parent != nullptr) {
        this->parent->Update(visit_add, win_times);
    }
    // 更新UCB的值-----更正：由于遍历顺序原因，UCB更新应当在遍历完子节点之后
    //this->UCB(2);
}


std::vector<int>* find_place(const Node *node) {
    // 找到某一个结点上可以落子的所有位置
    auto *place = new std::vector<int>;
    for (int i = 0; i < node->state->chess->Length*node->state->chess->Length; i++) {
        if (node->state->chess->mat[i] == -1) place->push_back(i);
    }
    return place;
}

// void rollout(Node *node, const std::vector<int> *place, int roll_times=1024) {
//     // 对某一个结点进行rollout，在一定次数以内随机落子
//     float  times=0, win_times=0;
//     if (node->state->Lear == 1) {
//         mut.lock();
//         // 对模拟结点进行反向传播
//         node->Update(1.0, 1.0);
//         mut.unlock();
//         delete place;
//         return;
//     }
//     std::srand(static_cast<unsigned int>(std::time(nullptr)));
//
//     while(roll_times > 0) {
//         times++;
//         roll_times--;
//
//         // 每次棋局 创建结点与落子坐标的副本
//         std::vector<int> temp_vector = *place;
//         const auto* temp = new Node(node->state);
//         while(temp->state->chess->Winner == -1 && !temp_vector.empty()) {
//             constexpr int min_value = 0;
//             const int max_value = static_cast<int>(temp_vector.size())-1;
//
//             // 计算vector内的随机index
//             const int index = min_value + std::rand() % (max_value - min_value + 1);
//
//             // 落子
//             const int y = (temp_vector[index] % temp->state->chess->Length);
//             const int x = (temp_vector[index] - y) / temp->state->chess->Length;
//             temp->state->chess->put(x, y);
//             // 丢弃走过的位置
//             temp_vector.erase(temp_vector.begin() + index);
//             // 判断
//             temp->state->chess->Judge();
//         }
//         if (temp->state->chess->Winner != -1) win_times = win_times + static_cast<float>(temp->state->chess->Winner);
//         delete temp;
//     }
//     // 对模拟结点进行反向传播
//     mut.lock();
//     node->Update(1, win_times/times);
//     mut.unlock();
//     delete place;
// }

int best_answer(const Node *node) {
    int answer = -1;
    float max_score = 0;
    for (int i = 0; i < node->Child.size(); i++) {
        if (node->Child[i]->score > max_score) {
            answer = i;
            max_score = node->Child[i]->score;
        }
    }
    return node->Child[answer]->state->chess->PlaceIndexLast;
}


int MCTS_search(const Chess* root_chess, const float time_limit) {
    // 初始化答案变量

    // 初始化状态根结点，并初次拓展与模拟，设置一个用于查找的temp指针
    auto root = Node(new State(root_chess));
    root.Expand();
    roll_and_update_GPU(&root, ROLL_TIMES);
    Node *temp = &root;
    // 在时间范围内进行迭代优化,每次从根开始查找
    const clock_t start = clock();
    clock_t end = clock();
    while (end - start <= time_limit) {
        temp = temp->Choice();
        if (temp == nullptr) break;  // 所有子结点都已经完全展开，收敛
        // 存在未完全展开的子结点，不断展开到未曾展开的子节点
        while (temp->Expended) {
            // 如果已经被展开了，就继续向下寻找未展开的子节点
            temp = temp->Choice();
            if (temp == nullptr) goto end;  // 在搜索时达到一个胜负点，说明此时已经收敛
        }
        temp->Expand();
        roll_and_update_GPU(temp, ROLL_TIMES);
        end = clock();
    }
    end:
    const int answer = best_answer(&root);
    return answer;
}

// void roll_thread(Node *Child,const int roll_times) {
//     rollout(Child, find_place(Child), roll_times);
// }
//
void UCB_thread(Node *child) {
    child->UCB();
}
//
// void roll_and_update(Node* root,int roll_times) {
//     std::vector<std::thread> pool;
//     pool.reserve(root->Child.size());
//
//     for(auto & i : root->Child) {
//         pool.emplace_back(roll_thread, i, roll_times);
//     }
//     for(int i = 0; i < root->Child.size(); i++) {
//         pool[i].join();
//     }
//
//     for(int i = 0; i < root->Child.size(); i++) {
//         pool[i] = std::thread(UCB_thread, root->Child[i]);
//     }
//     for(int i = 0; i < root->Child.size(); i++) {
//         pool[i].join();
//     }
// };

// 以下为GPU版本的函数
void roll_GPUthread(Node* child, const int roll_times) {
    std::vector<int>* place = find_place(child);
    const int place_eles = int(place->size());
    const auto place_arr = new int[place->size()];
    std::copy(place->begin(), place->end(), place_arr);

    RollOutGPU(child, child->state->chess->Length, place_arr, place_eles, roll_times);
    delete place;
    delete[] place_arr;
}


void roll_and_update_GPU(Node* root, int roll_times) {
    std::vector<std::thread> pool;
    pool.reserve(root->Child.size());
    for (auto & i : root->Child) {
            pool.emplace_back(roll_GPUthread, i, roll_times);
        }
    for (int i = 0; i < root->Child.size(); i++) {
        pool[i].join();
    }

    for (int i = 0; i < root->Child.size(); i++) {
        pool[i] = std::thread(UCB_thread, root->Child[i]);
    }
    for (int i = 0; i < root->Child.size(); i++) {
        pool[i].join();
    }
};