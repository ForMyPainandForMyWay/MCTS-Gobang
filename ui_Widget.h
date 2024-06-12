/********************************************************************************
** Form generated from reading UI file 'Widget.ui'
**
** Created by: Qt User Interface Compiler version 6.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H
#include "MyChessWidget.h"
#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Form
{
public:
    QWidget *verticalLayoutWidget;
    QVBoxLayout *layout;
    QVBoxLayout *verticalLayout;
    QPushButton *start_game;
    QLabel *who_put_show;
    QLabel *who_win_show;
    QPushButton *reput;
    QPushButton *reset_game;
    QComboBox *game_mode;
    MyChessWidget *widget;

    void setupUi(QWidget *Form)
    {
        if (Form->objectName().isEmpty())
            Form->setObjectName("Form");
        Form->resize(486, 363);
        verticalLayoutWidget = new QWidget(Form);
        verticalLayoutWidget->setObjectName("verticalLayoutWidget");
        verticalLayoutWidget->setGeometry(QRect(350, 20, 111, 301));
        layout = new QVBoxLayout(verticalLayoutWidget);
        layout->setSpacing(0);
        layout->setObjectName("layout");
        layout->setSizeConstraint(QLayout::SetDefaultConstraint);
        layout->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName("verticalLayout");
        start_game = new QPushButton(verticalLayoutWidget);
        start_game->setObjectName("start_game");
        start_game->setMinimumSize(QSize(0, 50));
        start_game->setCursor(QCursor(Qt::PointingHandCursor));

        verticalLayout->addWidget(start_game);

        who_put_show = new QLabel(verticalLayoutWidget);
        who_put_show->setObjectName("who_put_show");
        who_put_show->setMaximumSize(QSize(16777215, 40));
        QFont font;
        font.setPointSize(13);
        who_put_show->setFont(font);
        who_put_show->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(who_put_show);

        who_win_show = new QLabel(verticalLayoutWidget);
        who_win_show->setObjectName("who_win_show");
        who_win_show->setMaximumSize(QSize(16777215, 40));
        who_win_show->setFont(font);
        who_win_show->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(who_win_show);


        layout->addLayout(verticalLayout);

        reput = new QPushButton(verticalLayoutWidget);
        reput->setObjectName("reput");
        reput->setMinimumSize(QSize(0, 50));
        reput->setCursor(QCursor(Qt::PointingHandCursor));

        layout->addWidget(reput);

        reset_game = new QPushButton(verticalLayoutWidget);
        reset_game->setObjectName("reset_game");
        reset_game->setMinimumSize(QSize(0, 50));
        reset_game->setCursor(QCursor(Qt::PointingHandCursor));

        layout->addWidget(reset_game);

        game_mode = new QComboBox(verticalLayoutWidget);
        game_mode->addItem(QString());
        game_mode->addItem(QString());
        game_mode->setObjectName("game_mode");
        game_mode->setMinimumSize(QSize(0, 30));
        game_mode->setCursor(QCursor(Qt::OpenHandCursor));
        game_mode->setMouseTracking(false);

        layout->addWidget(game_mode);

        widget = new MyChessWidget(Form);
        widget->setObjectName("widget");
        widget->setGeometry(QRect(20, 20, 300, 300));

        retranslateUi(Form);

        game_mode->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Form);
    } // setupUi

    void retranslateUi(QWidget *Form)
    {
        Form->setWindowTitle(QCoreApplication::translate("Form", "我要成为 五子棋高手", nullptr));
        start_game->setText(QCoreApplication::translate("Form", "\345\274\200\345\247\213\346\270\270\346\210\217", nullptr));
        who_put_show->setText(QCoreApplication::translate("Form", "\350\257\267\351\273\221\346\243\213\350\220\275\345\255\220", nullptr));
        who_win_show->setText(QString());
        reput->setText(QCoreApplication::translate("Form", "\346\202\224\346\243\213", nullptr));
        reset_game->setText(QCoreApplication::translate("Form", "\351\207\215\347\275\256\346\270\270\346\210\217", nullptr));
        game_mode->setItemText(0, QCoreApplication::translate("Form", "PVP", nullptr));
        game_mode->setItemText(1, QCoreApplication::translate("Form", "PVE", nullptr));

#if QT_CONFIG(tooltip)
        game_mode->setToolTip(QCoreApplication::translate("Form", "\351\200\211\346\213\251\346\270\270\346\210\217\346\250\241\345\274\217", nullptr));
#endif // QT_CONFIG(tooltip)
        game_mode->setCurrentText(QCoreApplication::translate("Form", "PVP", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Form: public Ui_Form {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
