#pragma once

#include <QWidget>
//#include "ui_MyChessWidget.h"
#include "stack.h"
#include <deque>
#include <Game.h>

class MyChessWidget : public QWidget 
{  
	Q_OBJECT

public:
	int gridSize = 14; // 五子棋的格子数量  
	int margin = 10; // 边框与格子之间的间距
	int cellSize = 0;  // 每个格子的尺寸
	//std::deque<int>* step;  // 走过区域的栈，由Game类传入
	Stack_M<int>* step;
	Game* game = nullptr;
	MyChessWidget(QWidget* parent = nullptr);
	~MyChessWidget();
private:
	void paintEvent(QPaintEvent* event);  // 重载绘制函数
	void mousePressEvent(QMouseEvent* event);  // 重载点击函数
};
