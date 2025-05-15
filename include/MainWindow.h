#pragma once
#include "ui_Widget.h"
#include "Game.h"
#include <QtWidgets/QWidget>
#define PVP 0
#define PVE 1

class MainWindow final : public QWidget
{
	Q_OBJECT

public:
    Game* game;
    bool IsGaming;  // 正在游戏
    //bool endgame;  // 游戏结束
    bool AIisThinking;  // 正在搜索
    int GameMode = PVP;  // 游戏模式
    explicit MainWindow(Game* game, QWidget* parent = nullptr);
    ~MainWindow() override;
    void ShowWhoPut() const;  // 展示该谁落子
    void ShowWhoWin() const;  // 展示谁赢了
    void EndGame();  // 结束游戏

private:
    Ui::Form* ui;

private slots:
    void StartGame();  // 开始游戏按键
    void ReSetGame();  // 重置游戏按键
    void RePut();  // 悔棋
    void ReSetGameMode(); // 设置游戏模式
};
