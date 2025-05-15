#pragma once

#include <QWidget>
#include "Game.h"

class MyChessWidget final : public QWidget
{  
	Q_OBJECT

public:
	int GridSize = 14; // 五子棋的格子数量
	int Margin = 10; // 边框与格子之间的间距
	int CellSize = 0;  // 每个格子的尺寸
	Stack_MARR<int>* Step;  // 走过区域的栈，由Game类传入
	Game* game = nullptr;
	explicit MyChessWidget(QWidget* parent = nullptr);
	~MyChessWidget() override;
private slots:
	void paintEvent(QPaintEvent* event) override;  // 重载绘制函数
	void mousePressEvent(QMouseEvent* event) override;  // 重载点击函数
};
