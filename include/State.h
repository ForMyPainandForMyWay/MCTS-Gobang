#ifndef STATE_H
#define STATE_H


#include "Chess.h"
class State {
public:
    Chess *chess = nullptr;  // 某状态对应的棋盘，不保证已经裁决过
    int leaf = -1;  // 是否为叶结点？取值表示对应哪一方的叶节点
    std::vector<int> workable_place;  // 某状态对应的可行落子域

    State(Chess *chess);
    State(Chess *chess, int x_, int y_);  // 用于子结点的构造函数
    ~State();
    //Chess* next_state(int x, int y) const;
};
#endif //STATE_H
