#pragma once
#include "ui_Widget.h"
#include "Game.h"
#include <QtWidgets/QWidget>
#define PVP 0
#define PVE 1

class MAINWINDOW : public QWidget
{
	Q_OBJECT

public:
    Game* game;
    bool gamming;  // 正在游戏
    //bool endgame;  // 游戏结束
    bool AIisThinking;  // 正在搜索
    int game_mode = PVP;  // 游戏模式
    explicit MAINWINDOW(Game* game, QWidget* parent = nullptr);
    ~MAINWINDOW() override;
    void show_who_put();  // 展示该谁落子
    void show_who_win();  // 展示谁赢了
    void end_game();  // 结束游戏

private:
    Ui::Form* ui;

private slots:
    void start_game();  // 开始游戏按键
    void reset_game();  // 重置游戏按键
    void reput();  // 悔棋
    void reset_game_mode(); // 设置游戏模式
};
