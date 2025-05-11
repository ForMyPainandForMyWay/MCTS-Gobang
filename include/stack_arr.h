//
// Created by yyd on 25-5-11.
//

#ifndef STACK_ARR_H
#define STACK_ARR_H
#include <iostream>
#include <ostream>
#include <vector>
// 顺序栈的实现
template <typename T>
class Stack_MARR {
public:
    std::vector<T> *stack;  // 顺序栈
    int size_;

    Stack_MARR() : stack(new std::vector<T>), size_(0) {}

    ~Stack_MARR() {
        delete this->stack;
        this->size_ = 0;
    }

    // 压栈函数
    void push_back(T value) {
        this->stack->push_back(value);
        ++this->size_;
    }

    // 出栈函数
    void pop_back() {
        this->stack->pop_back();
        --this->size_;
    }

    // 返回栈顶数据
    T back() {
        return not empty() ? stack->back() : T();
    }

    // 判空
    bool empty() {
        return not this->size_;
    }

    // 获取大小
    int size() const
    {
        return this->size_;
    }

    // 清空
    void clear() {
        this->stack->clear();
        this->size_ = 0;
    }

    // 获取index位置处的数据
    T at(const int index) {
        return 0 < index < size_ ? (*this->stack)[index] : T();
    }
};

#endif //STACK_ARR_H
