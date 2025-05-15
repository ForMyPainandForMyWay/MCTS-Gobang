#include "MyChessWidget.h"

#include <QPainter>
#include <QMouseEvent>
#include <qmessagebox.h>
#include <MainWindow.h>

MyChessWidget::MyChessWidget(QWidget *parent): QWidget(parent), Step(nullptr){}

MyChessWidget::~MyChessWidget()
{
    this->Step = nullptr;
    this->game = nullptr;
}


void MyChessWidget::paintEvent(QPaintEvent* event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);   //边的反锯齿绘制
    painter.setRenderHint(QPainter::SmoothPixmapTransform);   //用平滑的pixmap变换算法

    // 获取边界, 绘制外框线
    const QRect outerRect = this->rect();
    painter.setPen(QPen(Qt::gray, 2));
    painter.setBrush(Qt::white);
    painter.drawRect(outerRect);

    // 绘制内框线
    const QRect innerRect = outerRect.adjusted(10, 10, -10, -10);
    painter.setPen(QPen(Qt::gray, 1));
    painter.drawRect(innerRect);

    this->CellSize = (outerRect.width() - 2 * this->Margin) / this->GridSize;

    painter.setPen(QPen(Qt::black, 0.5));
    for (int i = 0; i <= GridSize; ++i) {
        // 绘制水平线  
        const int x1 = innerRect.left() + i * CellSize;
        painter.drawLine(x1, innerRect.top(), x1, innerRect.bottom());
        // 绘制垂直线  
        const int y1 = innerRect.top() + i * CellSize;
        painter.drawLine(innerRect.left(), y1, innerRect.right(), y1);
    }

    // 绘制棋子，设置边框颜色
    int place = 0, y = 0, x = 0;  // y = place % 15, x = (place - y) / 15;
    painter.setPen(QPen(QColor(0, 0, 0), 1));

    for (int i = 0; i < this->Step->size(); i++) {
        place = this->Step->at(i);
        y = place % 15;
        x = (place - y) / (this->GridSize + 1);

        painter.setBrush(QColor(255 * (i%2), 255 * (i%2), 255 * (i%2)));
        painter.drawEllipse((innerRect.left() + y * CellSize) - 8, (innerRect.top() + x * CellSize) - 8, 16, 16);
    }
    // 绘制最后落子的棋子边框
    if (this->Step->size() != 0) {
        const QRect rect((innerRect.left() + y * CellSize) - 9, (innerRect.top() + x * CellSize) - 9, 18, 18);
        painter.setPen(QPen(Qt::red, 1));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(rect);
    }
}

void MyChessWidget::mousePressEvent(QMouseEvent* event) {
    auto* parentWidget = qobject_cast<MainWindow*>(parent());
    if (event->button() == Qt::LeftButton) {
        // 如果正在游戏且没有在搜索
        if (parentWidget->IsGaming and not parentWidget->AIisThinking) {
            const QRect innerRect = this->rect().adjusted(10, 10, -10, -10);
            const QPoint mouse_pos = event->pos();
            int x = mouse_pos.x(); // 获取鼠标横坐标，对应列数y

            if (int y = mouse_pos.y(); innerRect.contains(x, y)) {
                // int y_down = 0;
                // int x_right = 0;
                // int y_up = 0;
                // int x_left = 0;
                const int x_left = x - (x - 10) % this->CellSize;
                const int x_right = x_left + this->CellSize;

                const int y_up = y - (y - 10) % this->CellSize;
                const int y_down = y_up + this->CellSize;


                if (x - x_left > x_right - x) x = x_right / this->CellSize;
                else x = x_left / this->CellSize;

                if (y - y_up > y_down - y) y = y_down / this->CellSize;
                else y = y_up / this->CellSize;

                // 尝试落子
                if (bool done = this->game->Put(y, x)) {
                    this->repaint();
                    parentWidget->ShowWhoPut();
                    // 当点击方胜利时，展示并锁死局面
                    if (this->game->chess->Winner != -1) {
                        parentWidget->ShowWhoWin();
                        parentWidget->EndGame();
                        return;
                    }
                    // 否则，若为PVE模式，启动搜索
                    if (parentWidget->GameMode) {
                        parentWidget->AIisThinking = true;
                        this->setEnabled(false);
                        this->game->Search();
                        parentWidget->AIisThinking = false;
                        this->setEnabled(true);
                        this->update();
                        parentWidget->ShowWhoPut();
                        // 当AI胜利时，展示并锁死局面
                        if (this->game->chess->Winner != -1) {
                            parentWidget->ShowWhoWin();
                            parentWidget->EndGame();
                            return;
                        }
                    }

                    return;
                }

                // 落子失败的警告信息
                QMessageBox msgBox;
                msgBox.setWindowTitle("Tips");
                msgBox.setText("无法在该位置落子");
                msgBox.setIcon(QMessageBox::Warning);
                msgBox.exec();
            }
        }
    }
}