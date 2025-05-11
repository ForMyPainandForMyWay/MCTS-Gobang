#pragma once
#include <cuda_runtime.h>
#include "Node.h"

__device__ int transform_GPU(int x, int y, int chess_length);
__device__ int judge_GPU(const int* mat, int input_place, int chess_len, int player);
__global__ void roll_paralell(const int* mat,const int* step_nums,const int* place,int* win_times,const int *chess_len,const int *player_now);
void rollout_GPU(Node* node, int mat_size, const int* place, int place_eles, int roll_times);
