#include "Game.h"
#include "Node.h"
#define SIZE 15
#define METHOD 5
#define TIME_LIMIT 300000
Game::Game() {
    this->chess = new Chess(SIZE, METHOD);
}

Game::~Game() {
    delete this->chess;
    this->chess = nullptr;
}

// void Game::start_game() {
//     while(this->chess->Winner == -1) {
//         int x, y;
//         std::cin >> x >> y;
//         while (!this->chess->Put(x, y)) std::cin >> x >> y;;
//         this->step.push_back(this->chess->PlaceTransform(x, y));
//         //this->chess->show_mat();
//         this->chess->Judge();
//
//         if(this->chess->Winner != -1) {
//             this->ReSetGame();
//             continue;
//         }
//
//         this->Search();
//         //std::cin >> x >> y;
//         //while (!this->chess->Put(x, y)) std::cin >> x >> y;
//         //this->step.push(this->chess->PlaceTransform(x, y));
//         //this->chess->show_mat();
//         //this->chess->Judge();
//         //if(this->chess->Winner != -1) {
//         //    std::cout<<"Winner: "<<this->chess->Winner<<std::endl;
//         //    this->ReSetGame();
//         //}
//     }
// }

bool Game::Put(const int x, const int y){
    const bool done = this->chess->put(x, y);
    if (done) { 
        this->step.push_back(x * this->chess->Length + y);
        this->chess->Judge();
    }
    return done;
}

void Game::Rest(const int times) {
    if (this->step.size() < times) return;
    for (int i = 0; i < times; i++) {
        try {
            const int last = this->step.back();
            this->chess->mat[last] = -1;
            this->step.pop_back();
        }
        catch (...) {
            // std::cout << "Wrong: est error" << std::endl;
        }
    }
}

void Game::ReSetGame() {
    delete this->chess;
    this->chess = new Chess(SIZE, METHOD);
    this->step.Clear();
}

int Game::Search() {
    const int next = MCTS_search(this->chess, TIME_LIMIT);
    const int y = next % this->chess->Length;
    const int x = (next - y) / this->chess->Length;
    return this->Put(x, y);
}



