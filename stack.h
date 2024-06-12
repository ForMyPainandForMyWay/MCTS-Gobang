#pragma once
//#include <iostream>  

// 链栈结点的实现
template <typename T>
class NodeforS {
public:
    T data;
    NodeforS<T>* prev;  // 前驱
    NodeforS<T>* next;  // 后继

    NodeforS(T value) : data(value), prev(nullptr), next(nullptr) {};
    virtual ~NodeforS() {};
};


// 链栈的实现
template <typename T>
class Stack_M {
public:
    NodeforS<T>* head;  // 头节点为栈底
    NodeforS<T>* tail;  // 尾结点为栈顶
    int size_;

    Stack_M() : head(nullptr), tail(nullptr), size_(0) {}

    ~Stack_M() {
        this->deleteAllNodes();
    }

    // 压栈函数
    void push_back(T value) {
        NodeforS<T>* newNode = new NodeforS<T>(value);
        if (tail == nullptr) {
            this->head = this->tail = newNode;
        }
        else {
            newNode->prev = tail;
            this->tail->next = newNode;
            this->tail = newNode;
        }
        this->size_++;
    }

    // 出栈函数
    void pop_back() {
        if (this->tail == nullptr) {
            //std::cerr << "Stack is empty, cannot pop." << std::endl;
            return;
        }
        NodeforS<T>* temp = this->tail;
        if (this->head == this->tail) {
            this->head = this->tail = nullptr;
        }
        else {
            this->tail = this->tail->prev;
            this->tail->next = nullptr;
        }
        delete temp;
        this->size_--;
    }
    
    // 返回栈顶数据
    T back() {
        if (this->tail == nullptr) {
            //throw std::runtime_error("Stack is empty.");
        }
        return this->tail->data;
    }

    // 判空
    bool empty() {
        return this->size_ == 0;
    }

    // 获取大小
    int size() {
        return this->size_;
    }

    // 清空
    void clear() {
        deleteAllNodes();
    }

    // 获取index位置处的数据
    T at(size_t index) {
        if (index >= size_) {
            return 0;
        }
        NodeforS<T>* node = this->getNodeAt(index);
        return node->data;
    }

private:
    // 辅助函数，用于遍历到index位置并获取该位置结点
    NodeforS<T>* getNodeAt(size_t index) {
        if (index >= size_) {
            //throw std::out_of_range("Index out of range");
        }
        NodeforS<T>* current = head;
        for (size_t i = 0; i < index; ++i) {
            if (current == nullptr) {
                //throw std::out_of_range("Index out of range");
            }
            current = current->next;
        }
        return current;
    }

    // 辅助函数，用于删除所有结点
    void deleteAllNodes() {
        NodeforS<T>* current = head;
        while (current != nullptr) {
            NodeforS<T>* temp = current;
            current = current->next;
            delete temp;
        }
        this->head = this->tail = nullptr;
        this->size_ = 0;
    }
};