# include"Chess.h"
//#include <iomanip>
#include <iostream>
#define CROSS_BOUND 3
#define EMPTY (-1)
#define PLAYER_0 0  // 先手
#define PLAYER_1 1  // 后手（AI必为后手）
#define NO_MAN (-1)


Chess::Chess(int length, int method){
    //Player0 = 0,Player1 = 1
    this->method = method;
    this->length = length;
    this->mat = new int[length*length];
    for (int i = 0; i < length*length; i++) this->mat[i] = EMPTY;
    this->who_place_now = PLAYER_0;
    this->who_place_next = PLAYER_1;
    this->place_index_last = PLAYER_1;
    this->winner = NO_MAN;
}

Chess::Chess(const Chess *chess) {
    this->method = chess->method;
    this->length = chess->length;
    this->mat = new int[this->length * this->length];
    for (int i = 0; i < this->length * this->length; i++) {
        this->mat[i] = chess->mat[i];
    }
    this->who_place_now = chess->who_place_now;
    this->who_place_next = chess->who_place_next;
    this->place_index_last = chess->place_index_last;
    this->winner = chess->winner;
}

Chess::~Chess() {
    delete[] this->mat;
    this->mat = nullptr;
}

bool Chess::put(int x, int y) {
    int check_result = check(x, y);
    if (check_result == EMPTY) {
        this->mat[this->place_transform(x, y)] = this->who_place_now;
        this->who_place_now = this->who_place_next;
        this->who_place_next = (this->who_place_next + 1) % 2;
        this->place_index_last = this->place_transform(x, y);
        return true;
    }
    //std::cout << "Wrong Massage: place has been occupy or place is illegal." <<std::endl;
    return false;
}

int Chess::check(int x, int y) const {
    if (x >= this->length || y >= this->length || x < 0 || y < 0) return CROSS_BOUND;
    return this->mat[this->place_transform(x, y)];
}

int Chess::place_transform(int x, int y) const {
    return x * this->length + y;
}


int Chess::judge() {
    int x, y;
    int result_x=0,result_y=0, result_z=0, result_w=0;
    y = this->place_index_last % this->length;
    x = (this->place_index_last - y) / this->length;
    for (int i = -this->method+1; i <= this->method-1; i++) {
        // 逐行检查列
        if ((this->check(x + i, y) != CROSS_BOUND) and (this->check(x + i, y) == this->who_place_next)) result_y++;
        else result_y = 0;

        // 逐列检查行
        if ((this->check(x, y + i) != CROSS_BOUND) and (this->check(x, y + i) == this->who_place_next)) result_x++;
        else result_x = 0;

        // 检查主对角线
        if ((this->check(x + i, y + i) != CROSS_BOUND) and (this->check(x + i, y + i) == this->who_place_next)) result_z++;
        else result_z = 0;

        // 检查另一条对角线
        if ((this->check(x - i, y + i) != CROSS_BOUND) and (this->check(x - i, y + i) == this->who_place_next)) result_w++;
        else result_w = 0;

        if (result_x == this->method or result_y == this->method or result_z == this->method or result_w == this->method) {
            this->winner = this->who_place_next;
            return this->who_place_next;
        };
    }
    return NO_MAN;
}

// 以下展示函数已废除
/*
void Chess::show_mat() const {
    for (int i = 0; i < this->length; i++) {
        for (int j = 0; j < this->length; j++) {
            if(this->mat[this->place_transform(i, j)] == 1) std::cout <<std::setw(2) << "X" << std::setw(2);
            else if(this->mat[this->place_transform(i, j)] == 0) std::cout <<std::setw(2) << "O" << std::setw(2);
            else if(this->mat[this->place_transform(i, j)] == -1) std::cout <<std::setw(2) << " " << std::setw(2);
        }
        std::cout <<  std::endl;
    }
}*/



