#ifndef CHESS_H
#define CHESS_H

class Chess {
public:
    int Method = 5;  // 默认五子棋
    int Length=15;  // 棋盘大小
    int *mat = nullptr;  //棋盘矩阵
    int WhoPlaceNow=0;  //现在需要落子的对象
    int WhoPlaceNext=0;  //下一个落子的对象，也是上一个落子的对象
    int PlaceIndexLast=0;  //上一次落子的位置
    int Winner = -1;
    Chess(int length, int method);
    explicit Chess(const Chess *chess);
    ~Chess();
    [[nodiscard]] int PlaceTransform(int x, int y) const;
    bool put(int x, int y);
    [[nodiscard]] int check(int x, int y) const;
    int Judge();
    void SwapWhoPlace();
    //int judge_dfs();
    //void show_mat() const;
};

#endif //CHESS_H
