#ifndef MAINWINDOW_CPP
#define MAINWINDOW_CPP

#include "MainWindow.h"
#include <QtWidgets/QWidget>
#include <QMessageBox>
#define PVP 0
#define PVE 1

MainWindow::MainWindow(Game *game, QWidget* parent):
    QWidget(parent),
    ui(new Ui::Form)
{
    ui->setupUi(this);
    this->game = game;
    this->IsGaming = false;
    //this->endgame = true;  // 初始状态(结束游戏)不允许落子(已废弃)
    this->AIisThinking = false;
    this->GameMode = PVP;  // 初始游戏模式为PVP
    this->ui->widget->Step = &this->game->step;  // 将step导入绘图区窗口
    this->ui->widget->game = this->game;

    this->ui->widget->setEnabled(false);
    this->ui->reput->setEnabled(false);
    this->ui->reset_game->setEnabled(false);
    this->ui->who_put_show->setText(QString());
    
    connect(ui->start_game, &QPushButton::clicked, this, &MainWindow::StartGame);
    connect(ui->reset_game, &QPushButton::clicked, this, &MainWindow::ReSetGame);
    connect(ui->reput, &QPushButton::clicked, this, &MainWindow::RePut);
    connect(ui->game_mode, &QComboBox::currentTextChanged, this, &MainWindow::ReSetGameMode);
}

MainWindow::~MainWindow(){
    delete ui;
    delete game;
    ui = nullptr;
    game = nullptr;

}

// 开始游戏
void MainWindow::StartGame() {
    this->ReSetGame();
    this->IsGaming = true;
    //this->endgame = false;
    this->AIisThinking = false;
       
    //封锁控件
    this->ui->start_game->setEnabled(false);
    this->ui->widget->setEnabled(true);
    this->ui->reput->setEnabled(true);
    this->ui->reset_game->setEnabled(true);
    this->ui->game_mode->setEnabled(false);
    this->ShowWhoPut();
};


// 重置游戏
void MainWindow::ReSetGame() {
    if (this->AIisThinking) return;
    // 重置游戏状态
    this->IsGaming = false;
    this->AIisThinking = false;
    //this->endgame = false;

    // 重置棋盘,封锁或启用控件
    this->game->ReSetGame();
    this->ui->start_game->setEnabled(true);
    this->ui->widget->setEnabled(true);
    this->ui->widget->update();
    this->ui->game_mode->setEnabled(true);

    this->ui->reput->setEnabled(false);
    this->ui->reset_game->setEnabled(false);


    // 重置文字显示
    this->ui->who_put_show->setText(QString());
    this->ui->who_win_show->setText(QString());

};

// 悔棋
void MainWindow::RePut() {
    if (this->game->step.size() != 0) {
        this->game->Rest(this->GameMode + 1);
        // this->game->chess->WhoPlaceNext(this->game->Step);
        if (this->GameMode == PVP) this->game->chess->SwapWhoPlace();
        // this->game->chess->SwapWhoPlace();

        this->ui->widget->update();
        this->ShowWhoPut();
        this->ShowWhoWin();

        this->ui->widget->setEnabled(true);
        this->IsGaming = true;
        //this->endgame = false;
        this->AIisThinking = false;
    }
    else {
        // 设置提示消息
        QMessageBox msgBox;

        // 设置窗口标题,内容,图标
        msgBox.setWindowTitle("Tips");
        msgBox.setText("无法悔棋");
        msgBox.setIcon(QMessageBox::Warning);  

        // 显示对话框  
        msgBox.exec();
    }
};

// 模式切换
void MainWindow::ReSetGameMode() {
    // 当AI搜索时，阻止切换模式(已废弃，不会被调用)
    if (AIisThinking) {
        this->ui->game_mode->setCurrentIndex(PVE);
        return;
    };
    // 正常切换
    this->GameMode = this->ui->game_mode->currentIndex();
    this->ReSetGame();
};

// 展示哪一方落子，设置文字提示
void MainWindow::ShowWhoPut() const
{
    // Player_0=0 对应黑子，Player_1=1 对应白子
    if (this->game->chess->WhoPlaceNow) {
        this->ui->who_put_show->setText("请白棋落子");
    }
    else this->ui->who_put_show->setText("请黑棋落子");
};

// 只有有人胜利时才能调用，展示哪一方胜利，设置文字提示
void MainWindow::ShowWhoWin() const
{
    if (this->game->chess->Winner == -1) {
        this->ui->who_win_show->setText(QString());
        return;
    };
    if (this->game->chess->Winner == 0) this->ui->who_win_show->setText("黑棋胜利✌");
    else if (this->game->chess->Winner == 1) this->ui->who_win_show->setText("白棋胜利✌");
    this->ui->who_put_show->setText(QString());
};


// 结束局面，只有某一方胜利时才可以调用
void MainWindow::EndGame() {
    this->IsGaming = false;
    //this->endgame = true;
    this->AIisThinking = false;
    this->ui->widget->setEnabled(false);
    this->ui->start_game->setEnabled(false);
    this->ui->reput->setEnabled(false);
    this->ui->reset_game->setEnabled(true);
}
#endif // MAINWINDOW_CPP