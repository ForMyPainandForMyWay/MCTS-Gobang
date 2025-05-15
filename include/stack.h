#ifndef STACK_H
#define STACK_H

// 链栈结点的实现
template <typename T>
class NodeforS {
public:
    T data;
    NodeforS<T>* prev;  // 前驱
    NodeforS<T>* next;  // 后继

    explicit NodeforS(T value) : data(value), prev(nullptr), next(nullptr) {};
    virtual ~NodeforS() {};
};


// 链栈的实现
template <typename T>
class Stack_M {
public:
    NodeforS<T>* Head;  // 头节点为栈底
    NodeforS<T>* Tail;  // 尾结点为栈顶
    int Size;

    Stack_M() : Head(nullptr), Tail(nullptr), Size(0) {}

    ~Stack_M() {
        this->deleteAllNodes();
    }

    // 压栈函数
    void PushBack(T value) {
        auto* newNode = new NodeforS<T>(value);
        if (Tail == nullptr) {
            this->Head = this->Tail = newNode;
        }
        else {
            newNode->prev = Tail;
            this->Tail->next = newNode;
            this->Tail = newNode;
        }
        ++this->Size;
    }

    // 出栈函数
    void PopBack() {
        if (this->Tail == nullptr) {
            //std::cerr << "Stack is Empty, cannot pop." << std::endl;
            return;
        }
        const NodeforS<T>* temp = this->Tail;
        if (this->Head == this->Tail) {
            this->Head = this->Tail = nullptr;
        }
        else {
            this->Tail = this->Tail->prev;
            this->Tail->next = nullptr;
        }
        delete temp;
        --this->Size;
    }

    // 返回栈顶数据
    T Back() {
        if (this->Tail == nullptr) {
            //throw std::runtime_error("Stack is Empty.");
        }
        return this->Tail->data;
    }

    // 判空
    bool Empty() {
        return this->Size == 0;
    }

    // 获取大小
    [[nodiscard]] int size() const
    {
        return this->Size;
    }

    // 清空
    void Clear() {
        deleteAllNodes();
    }

    // 获取index位置处的数据
    T at(const int index) {
        if (index >= Size) {
            return 0;
        }
        NodeforS<T>* node = this->getNodeAt(index);
        return node->data;
    }

private:
    // 辅助函数，用于遍历到index位置并获取该位置结点
    NodeforS<T>* getNodeAt(const int index) {
        if (index >= Size) {
            //throw std::out_of_range("Index out of range");
        }
        NodeforS<T>* current = Head;
        for (int i = 0; i < index; ++i) {
            if (current == nullptr) {
                //throw std::out_of_range("Index out of range");
            }
            current = current->next;
        }
        return current;
    }

    // 辅助函数，用于删除所有结点
    void deleteAllNodes() {
        NodeforS<T>* current = Head;
        while (current != nullptr) {
            const NodeforS<T>* temp = current;
            current = current->next;
            delete temp;
        }
        this->Head = this->Tail = nullptr;
        this->Size = 0;
    }
};

#endif  // STACK_H