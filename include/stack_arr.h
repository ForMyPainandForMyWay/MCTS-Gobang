//
// Created by yyd on 25-5-11.
//

#ifndef STACK_ARR_H
#define STACK_ARR_H
#include <vector>
// 顺序栈的实现
template <typename T>
class Stack_MARR {
public:
    std::vector<T> *stack;  // 顺序栈
    int Size;

    Stack_MARR() : stack(new std::vector<T>), Size(0){}

    ~Stack_MARR() {
        delete this->stack;
        this->Size = 0;
    }

    // 压栈函数
    void push_back(T value) {
        this->stack->push_back(value);
        ++this->Size;
    }

    // 出栈函数
    void pop_back() {
        this->stack->pop_back();
        --this->Size;
    }

    // 返回栈顶数据
    T back() {
        return not empty() ? stack->back() : T();
    }

    // 判空
    bool empty() {
        return not this->Size;
    }

    // 获取大小
    [[nodiscard]] int size() const
    {
        return this->Size;
    }

    // 清空
    void Clear() {
        this->stack->clear();
        this->Size = 0;
    }

    // 获取index位置处的数据
    T at(const int index) {
        return 0 < index < Size ? (*this->stack)[index] : T();
    }
};

#endif //STACK_ARR_H
