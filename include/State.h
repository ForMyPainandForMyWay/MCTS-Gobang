#ifndef STATE_H
#define STATE_H


#include "Chess.h"
#include <vector>
class State {
public:
    Chess *chess = nullptr;  // 某状态对应的棋盘，不保证已经裁决过
    int Lear = -1;  // 是否为叶结点？取值表示对应哪一方的叶节点
    std::vector<int> WorkablePlace;  // 某状态对应的可行落子域

    explicit State(const Chess *chess);
    State(const Chess *chess, int x_, int y_);  // 用于子结点的构造函数
    ~State();
    //Chess* next_state(int x, int y) const;
};
#endif //STATE_H
