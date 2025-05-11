#include "Game.h"
#include "Node.h"
#include <iostream>
#define SIZE 15
#define METHOD 5
#define TIME_LIMIT 15000
Game::Game() {
    this->chess = new Chess(SIZE, METHOD);
}

Game::~Game() {
    delete this->chess;
    this->chess = nullptr;
}

void Game::start_game() {
    while(this->chess->winner == -1) {
        int x, y;
        std::cin >> x >> y;
        while (!this->chess->put(x, y)) std::cin >> x >> y;;
        this->step.push_back(this->chess->place_transform(x, y));
        //this->chess->show_mat();
        this->chess->judge();
        std::cout << this->chess->winner << std::endl;

        if(this->chess->winner != -1) {
            std::cout<<"winner: "<<this->chess->winner<<std::endl;
            this->reset_game();
            continue;
        }

        this->search();
        //std::cin >> x >> y;
        //while (!this->chess->put(x, y)) std::cin >> x >> y;
        //this->step.push(this->chess->place_transform(x, y));
        //this->chess->show_mat();
        //this->chess->judge();
        //if(this->chess->winner != -1) {
        //    std::cout<<"winner: "<<this->chess->winner<<std::endl;
        //    this->reset_game();
        //}
    }
}

bool Game::put(int x, int y){
    bool done = this->chess->put(x, y);
    if (done) { 
        this->step.push_back(x * this->chess->length + y); 
        this->chess->put(x, y);
        //this->chess->judge_dfs();
        this->chess->judge();
    }
    return done;
}

void Game::est(int times) {
    int last;
    if (this->step.size() < times) return;
    for (int i = 0; i < times; i++) {
        try {
            last = this->step.back();
            this->chess->mat[last] = -1;
            this->step.pop_back();
        }
        catch (...) {
            std::cout << "Wrong: est error" << std::endl;
        }
    }
}

void Game::reset_game() {
    delete this->chess;
    this->chess = new Chess(SIZE, METHOD);
    this->step.clear();
}

int Game::search() {
    int x, y;
    bool done;
    int next = MCTS_search(this->chess, TIME_LIMIT);
    y = next % this->chess->length;
    x = (next - y) / this->chess->length;

    done = this->put(x, y);
    return  0;
}



