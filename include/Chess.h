#ifndef CHESS_H
#define CHESS_H

#include <vector>
class Chess {
public:
    int method = 5;  // 默认五子棋
    int length=15;  // 棋盘大小
    int *mat = nullptr;  //棋盘矩阵
    int who_place_now=0;  //现在需要落子的对象
    int who_place_next=0;  //下一个落子的对象，也是上一个落子的对象
    int place_index_last=0;  //上一次落子的位置
    int winner = -1;
    Chess(int length, int method);
    Chess(const Chess *chess);
    ~Chess();
    int place_transform(int x, int y) const;
    bool put(int x, int y);
    int check(int x, int y) const;
    int judge();
    //int judge_dfs();
    //void show_mat() const;
};

#endif //CHESS_H
