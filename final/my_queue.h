#pragma once
#include <iostream>
#include "cuda_runtime.h"

struct Node {
	int index;
	int depth;
};

class My_Queue 
{
	public:
		__device__ My_Queue(Node* values, int capacity) {
			this->values = values;
			this->capacity = capacity;
			_front = 0;
			_rear = -1;
			counter = 0;
		}
		__device__ ~My_Queue() {}

		__device__ bool is_empty(void) {
			if (counter) return false;
			else return true;
		}
		__device__ bool is_full(void) {
			if (counter == capacity) return true;
			else return false;
		}

		__device__ void push(Node x) {
			_rear = (_rear + 1) % capacity;
			values[_rear] = x;
			counter++;
		}
		__device__ Node front() {
			return values[_front];
		}
		__device__ void pop() {
			_front = (_front + 1) % capacity;
			counter--;
		}
	private:
		int _front;// front index
		int _rear;// rear index
		int counter;
		int capacity;
		Node* values;
};
