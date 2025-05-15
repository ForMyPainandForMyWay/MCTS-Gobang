#include "Game.h"
#include "MainWindow.h"

using namespace std;

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    const auto game = new Game;
    MainWindow wid(game);
    wid.show();
    
    return QApplication::exec();
}