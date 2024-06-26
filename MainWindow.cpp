﻿#ifndef MAINWINDOW_CPP
#define MAINWINDOW_CPP

#include <QMessageBox>  
#include "ui_Widget.h"
#include "MainWindow.h"
#include <QPainter>
#include <QtWidgets/QWidget>
#include <Qobject>
#include <QMessageBox>
#include <iostream>
#define PVP 0
#define PVE 1

MAINWINDOW::MAINWINDOW(Game *game, QWidget* parent):
    QWidget(parent),
    ui(new Ui::Form)
{
    ui->setupUi(this);
    this->game = game;
    this->gamming = false;
    //this->endgame = true;  // 初始状态(结束游戏)不允许落子(已废弃)
    this->AIisThinking = false;
    this->game_mode = PVP;  // 初始游戏模式为PVP
    this->ui->widget->step = &this->game->step;  // 将step导入绘图区窗口
    this->ui->widget->game = this->game;

    this->ui->widget->setEnabled(false);
    this->ui->reput->setEnabled(false);
    this->ui->reset_game->setEnabled(false);
    this->ui->who_put_show->setText(QString());
    

    QObject::connect(ui->start_game, &QPushButton::clicked, this, &MAINWINDOW::start_game);
    QObject::connect(ui->reset_game, &QPushButton::clicked, this, &MAINWINDOW::reset_game);
    QObject::connect(ui->reput, &QPushButton::clicked, this, &MAINWINDOW::reput);
    QObject::connect(ui->game_mode, &QComboBox::currentTextChanged, this, &MAINWINDOW::reset_game_mode);
}

MAINWINDOW::~MAINWINDOW(){
    delete ui;
    delete game;
    ui = nullptr;
    game = nullptr;

}

// 开始游戏
void MAINWINDOW::start_game() {
    this->reset_game();
    this->gamming = true;
    //this->endgame = false;
    this->AIisThinking = false;
    
       
    //封锁控件
    this->ui->start_game->setEnabled(false);
    this->ui->widget->setEnabled(true);
    this->ui->reput->setEnabled(true);
    this->ui->reset_game->setEnabled(true);
    this->ui->game_mode->setEnabled(false);
    this->show_who_put();

};


// 重置游戏
void MAINWINDOW::reset_game() {
    if (this->AIisThinking) return;
    // 重置游戏状态
    this->gamming = false;
    this->AIisThinking = false;
    //this->endgame = false;

    // 重置棋盘,封锁或启用控件
    this->game->reset_game();
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
void MAINWINDOW::reput() {
    if (this->game->step.size() != 0) {
        this->game->est(this->game_mode + 1);
        this->ui->widget->update();
        this->show_who_put();
        this->show_who_win();

        this->ui->widget->setEnabled(true);
        this->gamming = true;
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
void MAINWINDOW::reset_game_mode() {
    // 当AI搜索时，阻止切换模式(已废弃，不会被调用)
    if (AIisThinking) {
        this->ui->game_mode->setCurrentIndex(PVE);
        return;
    };
    // 正常切换
    this->game_mode = this->ui->game_mode->currentIndex();
    this->reset_game();
};

// 展示哪一方落子，设置文字提示
void MAINWINDOW::show_who_put() {
    // Player_0=0 对应黑子，Player_1=1 对应白子
    if (this->game->chess->who_place_now) {
        this->ui->who_put_show->setText("请白棋落子");
    }
    else this->ui->who_put_show->setText("请黑棋落子");
};

// 只有有人胜利时才能调用，展示哪一方胜利，设置文字提示
void MAINWINDOW::show_who_win() {
    if (this->game->chess->winner == -1) {
        this->ui->who_win_show->setText(QString());
        return;
    };
    if (this->game->chess->winner == 0) this->ui->who_win_show->setText("黑棋胜利✌"); 
    else if (this->game->chess->winner == 1) this->ui->who_win_show->setText("白棋胜利✌");
    this->ui->who_put_show->setText(QString());
    return;
};


// 结束局面，只有某一方胜利时才可以调用
void MAINWINDOW::end_game() {
    this->gamming = false;
    //this->endgame = true;
    this->AIisThinking = false;
    this->ui->widget->setEnabled(false);
    this->ui->start_game->setEnabled(false);
    this->ui->reput->setEnabled(false);
    this->ui->reset_game->setEnabled(true);
}
#endif // MAINWINDOW_CPP