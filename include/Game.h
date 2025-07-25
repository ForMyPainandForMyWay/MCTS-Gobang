#ifndef GAME_H
#define GAME_H

#define AI 1
#define HUMAN 0
#include "Chess.h"
#include "stack_arr.h"


class Game {
public:
    Chess *chess = nullptr;
    int Player_0 = HUMAN;
    int Player_1 = AI;
    Stack_MARR<int> step;
    Game();
    ~Game();
    //void StartGame();  // 开始游戏，用于命令行窗口
    bool Put(int x, int y);
    void Rest(int times = 2);  // 不同模式的悔棋
    int Search();  // 蒙特卡洛树搜索
    void ReSetGame();  // 重置游戏
};
#endif //GAME_H
