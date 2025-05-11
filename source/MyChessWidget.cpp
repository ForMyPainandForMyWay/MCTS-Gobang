#include "MyChessWidget.h"
#include <QPainter>
#include <QMouseEvent>
#include <qmessagebox.h>
#include <MainWindow.h>

MyChessWidget::MyChessWidget(QWidget *parent)
	: QWidget(parent)
{
    this->step = nullptr;
}

MyChessWidget::~MyChessWidget()
{
    this->step = nullptr;
    this->game = nullptr;
}


void MyChessWidget::paintEvent(QPaintEvent* event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);   //边的反锯齿绘制
    painter.setRenderHint(QPainter::SmoothPixmapTransform);   //用平滑的pixmap变换算法

    // 获取边界, 绘制外框线
    QRect outerRect = this->rect();
    painter.setPen(QPen(Qt::gray, 2));
    painter.setBrush(Qt::white);
    painter.drawRect(outerRect);

    // 绘制内框线
    QRect innerRect = outerRect.adjusted(10, 10, -10, -10);
    painter.setPen(QPen(Qt::gray, 1));
    painter.drawRect(innerRect);

    this->cellSize = (outerRect.width() - 2 * this->margin) / this->gridSize;

    painter.setPen(QPen(Qt::black, 0.5));
    for (int i = 0; i <= gridSize; ++i) {
        // 绘制水平线  
        int x1 = innerRect.left() + i * cellSize;
        painter.drawLine(x1, innerRect.top(), x1, innerRect.bottom());
        // 绘制垂直线  
        int y1 = innerRect.top() + i * cellSize;
        painter.drawLine(innerRect.left(), y1, innerRect.right(), y1);
    }

    // 绘制棋子，设置边框颜色
    int place = 0, y = 0, x = 0;  // y = place % 15, x = (place - y) / 15;
    painter.setPen(QPen(QColor(0, 0, 0), 1));

    for (int i = 0; i < this->step->size(); i++) {
        place = this->step->at(i);
        y = place % 15;
        x = (place - y) / (this->gridSize + 1);

        painter.setBrush(QColor(255 * (i%2), 255 * (i%2), 255 * (i%2)));
        painter.drawEllipse((innerRect.left() + y * cellSize) - 8, (innerRect.top() + x * cellSize) - 8, 16, 16);
    }
    // 绘制最后落子的棋子边框
    if (this->step->size() != 0) {
        QRect rect((innerRect.left() + y * cellSize) - 9, (innerRect.top() + x * cellSize) - 9, 18, 18);
        painter.setPen(QPen(Qt::red, 1));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(rect);
    }
}

void MyChessWidget::mousePressEvent(QMouseEvent* event) {
    bool done = false;
    MAINWINDOW* parentWidget = qobject_cast<MAINWINDOW*>(parent());
    if (event->button() == Qt::LeftButton) {
        // 如果正在游戏且没有在搜索
        if (parentWidget->gamming and not parentWidget->AIisThinking) {
            const QRect innerRect = this->rect().adjusted(10, 10, -10, -10);
            const QPoint mouse_pos = event->pos();
            int x_left = 0, y_up = 0, x_right = 0, y_down = 0;
            int x = mouse_pos.x(); // 获取鼠标横坐标，对应列数y
            int y = mouse_pos.y(); // 获取鼠标纵坐标，对应行数x

            if (innerRect.contains(x, y)) {
                x_left = (x - ((x - 10) % this->cellSize));
                x_right = x_left + this->cellSize;

                y_up = (y - ((y - 10) % this->cellSize));
                y_down = y_up + this->cellSize;


                if (x - x_left > x_right - x) x = x_right / this->cellSize;
                else x = x_left / this->cellSize;

                if (y - y_up > y_down - y) y = y_down / this->cellSize;
                else y = y_up / this->cellSize;

                // 尝试落子
                done = this->game->put(y, x);
                if (done) {
                    this->repaint();
                    parentWidget->show_who_put();
                    // 当点击方胜利时，展示并锁死局面
                    if (this->game->chess->winner != -1) {
                        parentWidget->show_who_win();
                        parentWidget->end_game();
                        return;
                    }
                    // 否则，若为PVE模式，启动搜索
                    if (parentWidget->game_mode) {
                        parentWidget->AIisThinking = true;
                        this->setEnabled(false);
                        this->game->search();
                        parentWidget->AIisThinking = false;
                        this->setEnabled(true);
                        this->update();
                        parentWidget->show_who_put();
                        // 当AI胜利时，展示并锁死局面
                        if (this->game->chess->winner != -1) {
                            parentWidget->show_who_win();
                            parentWidget->end_game();
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