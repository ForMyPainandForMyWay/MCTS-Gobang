#include "Chess.h"

#define CROSS_BOUND 3
#define EMPTY (-1)
#define PLAYER_0 0  // 先手
#define PLAYER_1 1  // 后手（AI必为后手）
#define NO_MAN (-1)


Chess::Chess(const int length, const int method){
    //Player0 = 0,Player1 = 1
    this->Method = method;
    this->Length = length;
    this->mat = new int[length*length];
    for (int i = 0; i < length*length; i++) this->mat[i] = EMPTY;
    this->WhoPlaceNow = PLAYER_0;
    this->WhoPlaceNext = PLAYER_1;
    this->PlaceIndexLast = PLAYER_1;
    this->Winner = NO_MAN;
}

Chess::Chess(const Chess *chess) {
    this->Method = chess->Method;
    this->Length = chess->Length;
    this->mat = new int[this->Length * this->Length];
    for (int i = 0; i < this->Length * this->Length; i++) {
        this->mat[i] = chess->mat[i];
    }
    this->WhoPlaceNow = chess->WhoPlaceNow;
    this->WhoPlaceNext = chess->WhoPlaceNext;
    this->PlaceIndexLast = chess->PlaceIndexLast;
    this->Winner = chess->Winner;
}

Chess::~Chess() {
    delete[] this->mat;
    this->mat = nullptr;
}

bool Chess::put(const int x, const int y) {
    if (const int check_result = check(x, y); check_result == EMPTY) {
        this->mat[this->PlaceTransform(x, y)] = this->WhoPlaceNow;
        this->WhoPlaceNow = this->WhoPlaceNext;
        this->WhoPlaceNext = (this->WhoPlaceNext + 1) % 2;
        this->PlaceIndexLast = this->PlaceTransform(x, y);
        return true;
    }
    return false;
}

int Chess::check(const int x, const int y) const {
    if (x >= this->Length || y >= this->Length || x < 0 || y < 0) return CROSS_BOUND;
    return this->mat[this->PlaceTransform(x, y)];
}

int Chess::PlaceTransform(const int x, const int y) const {
    return x * this->Length + y;
}


int Chess::Judge() {
    int result_x=0,result_y=0, result_z=0, result_w=0;
    const int y = this->PlaceIndexLast % this->Length;
    const int x = (this->PlaceIndexLast - y) / this->Length;
    for (int i = -this->Method+1; i <= this->Method-1; i++) {
        // 逐行检查列
        if ((this->check(x + i, y) != CROSS_BOUND) and (this->check(x + i, y) == this->WhoPlaceNext)) result_y++;
        else result_y = 0;

        // 逐列检查行
        if ((this->check(x, y + i) != CROSS_BOUND) and (this->check(x, y + i) == this->WhoPlaceNext)) result_x++;
        else result_x = 0;

        // 检查主对角线
        if ((this->check(x + i, y + i) != CROSS_BOUND) and (this->check(x + i, y + i) == this->WhoPlaceNext)) result_z++;
        else result_z = 0;

        // 检查另一条对角线
        if ((this->check(x - i, y + i) != CROSS_BOUND) and (this->check(x - i, y + i) == this->WhoPlaceNext)) result_w++;
        else result_w = 0;

        if (result_x == this->Method or result_y == this->Method or result_z == this->Method or result_w == this->Method) {
            this->Winner = this->WhoPlaceNext;
            return this->WhoPlaceNext;
        };
    }
    return NO_MAN;
}

void Chess::SwapWhoPlace()
{
    const int tmp = this->WhoPlaceNow;
    this->WhoPlaceNow = this->WhoPlaceNext;
    this->WhoPlaceNext = tmp;
}


