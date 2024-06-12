#include"Game.h"
#include<QApplication>
#include"MainWindow.h"

using namespace std;

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    Game* game = new Game;
    MAINWINDOW wid(game);
    wid.show();
    
    return a.exec();
}