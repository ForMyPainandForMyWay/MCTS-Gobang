#ifndef GAME_H
#define GAME_H
#define AI 1
#define HUMAN 0
//#include <stack>
#include "stack.h"
#include <deque>
#include "Chess.h"

class Game {
public:
    Chess *chess = nullptr;
    int Player_0 = HUMAN;
    int Player_1 = AI;
    //std::deque<int> step;  // 存储走过的区域的栈
    Stack_M<int> step;
    Game();
    ~Game();
    void start_game();  // 开始游戏，用于命令行窗口
    bool put(int x, int y);
    void est(int times = 2);  // 不同模式的悔棋
    int search();  // 蒙特卡洛树搜索
    void reset_game();  // 重置游戏
};
#endif //GAME_H
