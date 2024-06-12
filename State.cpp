#include"State.h"

#include <iostream>
#define EMPTY (-1)

State::State(Chess *chess) {
    // 初始化状态的棋盘(拷贝一个)
    this->chess = new Chess(chess);

    int y = this->chess->place_index_last % this->chess->length;
    int x = (this->chess->place_index_last - y) / this->chess->length;

    // 初始化可行域
    for (int i = x-2; i <= x+2; i++) {
        for (int j = y-2; j <= y+2; j++) {
            if (this->chess->check(i, j) == EMPTY) {
                this->workable_place.push_back(this->chess->place_transform(i, j));
            }
        }
    }
}

State::~State() {
    delete this->chess;
    this->chess = nullptr;
}

// Chess* State::next_state(int x, int y) const {
//     auto *next = new Chess(this->chess);
//     next->put(x, y);
//     return next;
// }

State::State(Chess *chess, int x_, int y_) {
    // 主要用于子节点的构造函数，传入chess和可落子的位置，生成落子后的状态
    this->chess = new Chess(chess);
    // 落子
    this->chess->put(x_, y_);
    if (this->chess->judge() == -1) {
        // 当这个子节点不是叶结点时，初始化可行域
        for (int i = x_-2; i <= x_+2; i++) {
            for (int j = y_-2; j <= y_+2; j++) {
                if (this->chess->check(i, j) == EMPTY) {
                    this->workable_place.push_back(this->chess->place_transform(i, j));
                }
            }
        }
    }
    else {
        // 获胜后为叶节点，无需寻找可以落子的位置
        this->leaf = this->chess->winner;
    }
}