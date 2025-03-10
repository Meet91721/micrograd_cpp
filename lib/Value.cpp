#include "Value.h"
#include <_strings.h>
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <memory>

/* for printing the Value object (right now implemented by overloading the
 * ostream << operator)*/
std::ostream &operator<<(std::ostream &out_stream, Value &obj) noexcept {

	out_stream << obj.ptr->label << " = " << *(obj.ptr->data)
             << " | grad = " << *(obj.ptr->grad) << std::endl;
	return out_stream;
}

/* for printing the Value object (right now implemented by overloading the
 * ostream << operator)*/
std::ostream &operator<<(std::ostream &out_stream, Value &&obj) noexcept {

	out_stream << obj.ptr->label << " = " << *(obj.ptr->data)
             << " | grad = " << *(obj.ptr->grad) << std::endl;
	return out_stream;
}

/* operation definitions for Value objects */

void iterator(std::vector<Selector> &indexes, int depth, int &index, int ori_index, int box_cap, std::vector<std::pair<int,int>> &res, std::vector<int> &ori_shape){

	if(depth == indexes.size()){
		// view.ptr->data[index++] = original.ptr->data[ori_index];
		res.push_back({index++, ori_index});
		return;
	}
	box_cap /= ori_shape[depth];
	if(indexes[depth].isRange){
		for(int i = indexes[depth].indexes[0]; i < indexes[depth].indexes[1]; i++){
			int new_ori_index = ori_index + box_cap * i;
			iterator(indexes, depth + 1, index, new_ori_index, box_cap, res, ori_shape);
		}
	}else{
		for(int i = 0; i < indexes[depth].n_elements; i++){
			int new_ori_index = ori_index + box_cap * indexes[depth].indexes[i];
			std::cout << "Printing index here: " << indexes[depth].indexes[i] << " depth: " << box_cap << " " << new_ori_index << '\n';
			iterator(indexes, depth + 1, index, new_ori_index, box_cap, res, ori_shape);
		}
	}
}

std::vector<std::pair<int,int>> viewCreator(Value &original, Value &view, std::vector<Selector> &indexes){

	int n_elements = 1;
	for(int i: original.ptr->shape){
		n_elements *= i;
	}
	std::vector<std::pair<int,int>> res;
	int index = 0;
	iterator(indexes, 0, index, 0, n_elements, res, original.ptr->shape);
	return res;
}

Value Value::operator()(std::vector<Selector> &&indexes){

	assert(indexes.size() == this->ptr->shape.size() && "Incorrect shape demanded");
	std::vector<int>shape;
	for(int i = 0; i < indexes.size(); i++){
		indexes[i].isValid(this->ptr->shape[i]);
		shape.push_back(indexes[i].n_elements);
	}
	/*
	int i = 0;
	for(auto &index: indexes){
		index.isValid(this->ptr->shape[i++]);
		shape.push_back(index.n_elements);
	}
	*/
	// Projection p = Projection(indexes, this->ptr->data);
// Value(std::vector<int> _shape,std::string _label = "",std::vector<std::shared_ptr<valueData>> _children = {},std::string _op = "",std::function<void()> __backward = nullptr)
	Value out = Value(shape);
	std::vector<std::pair<int,int>> mappings = viewCreator(*this, out, indexes);
	std::cout << "Here we go:\n";
	/*
	for(int i = 0; i < mappings.size(); i++){
		std::cout << mappings[i].first << " " << mappings[i].second << '\n';
		out.ptr->data+mappings[i].first = :
	}*/
	return out;
}

// addition
Value Value::operator+(Value &other) {

	Value out = Value(other.ptr->shape);
	int size = out.ptr->total_size;
	std::transform(this->ptr->data, this->ptr->data + size, other.ptr->data,
                 out.ptr->data, [](double a, double b) { return a + b; });
	out.ptr->children = {this->ptr, other.ptr};
	out.ptr->op = "+";
	out.ptr->_backward = [w_ptr = std::weak_ptr<valueData>(out.ptr), size]() {
		std::shared_ptr<valueData> out_ptr = w_ptr.lock();
		for (int i = 0; i < size; i++) {

				out_ptr->children[0]->grad[i] += double(1.0) * out_ptr->grad[i];
				out_ptr->children[1]->grad[i] += double(1.0) * out_ptr->grad[i];
		}
	};
	return out;
}

// for supporting the Value + double
Value Value::operator+(double val) {

	Value val_obj = Value(val);
	return (*this + val_obj);
}

// for supporting the double + Value
Value operator+(double val, Value &other) { 

	return other + val; 
}

// multiplication
Value Value::operator*(Value &other) {

	Value out = Value(other.ptr->shape);
	int size = out.ptr->total_size;
	std::transform(this->ptr->data, this->ptr->data + size, other.ptr->data,
                 out.ptr->data, [](double a, double b) { return a * b; });
	out.ptr->label = "*";
	out.ptr->children = {this->ptr, other.ptr};
	std::shared_ptr<valueData> out_ptr = out.ptr;
	out.ptr->_backward = [w_ptr = std::weak_ptr<valueData>(out.ptr), size]() {
		std::shared_ptr<valueData> out_ptr = w_ptr.lock();
		for (size_t index = 0; index < size; index++) {
			out_ptr->children[0]->grad[index] +=
					(out_ptr->children[1]->data[index]) * (out_ptr->grad[index]);
			out_ptr->children[1]->grad[index] +=
					(out_ptr->children[0]->data[index]) * (out_ptr->grad[index]);
		}
	};
	return out;
}

Value Value::operator*(double val) {

	Value other = Value(val);
	return (*this) * (other);
}

Value operator*(double val, Value &other) { 

	return other * val; 
}


// substraction
Value Value::operator-(Value &other) {

	Value tmp = other * (-1);
	int size = tmp.ptr->total_size;
	return ((*this) + tmp);
}

Value Value::operator-(double val) {

	Value other = Value(val);
	return ((*this) - (other));
}

Value operator-(double val, Value &other) {

	Value tmp = Value(val);
	return tmp - other;
}


// division
Value Value::operator/(Value &other) {

	Value tmp = other ^ (-1);
	return ((*this) * tmp);
}

Value Value::operator/(double val) {

	Value other = Value(val);
	return ((*this) / other);
}

Value operator/(double val, Value &other) {

	Value tmp = Value(val);
	return tmp / other;
}


// power
Value Value::operator^(Value &other) {

	Value out = Value(other.ptr->shape);
	int size = out.ptr->total_size;
	std::transform(this->ptr->data, this->ptr->data + size, other.ptr->data,
			 out.ptr->data,
			 [](double a, double b) { return std::pow(a, b); });
	out.ptr->label = "^";
	out.ptr->children = {this->ptr, other.ptr};
	std::shared_ptr<valueData> out_ptr = out.ptr;
	out.ptr->_backward = [w_ptr = std::weak_ptr<valueData>(out.ptr), size]() {
		std::shared_ptr<valueData> out_ptr = w_ptr.lock();
		std::cout << "This address1: " << &out_ptr << " " << out_ptr << '\n';
		double *this_data = (out_ptr->children[0]->data);
		double *other_data = (out_ptr->children[1]->data);

		for (size_t index = 0; index < size; index++) {
			out_ptr->children[0]->grad[index] +=
				  (std::pow(this_data[index], other_data[index]) * other_data[index] /
				   this_data[index]) * out_ptr->grad[index];
			out_ptr->children[1]->grad[index] +=
				  (std::pow(this_data[index], other_data[index]) *
				   log(this_data[index])) * out_ptr->grad[index];
		}
	};
	return out;
}

Value Value::operator^(double val) {

	Value other = Value(val);
	return ((*this) ^ (other));
}

Value operator^(double val, Value &other) {

	Value tmp = Value(val);
	return tmp ^ other;
}

/*Functions and helpers for calculating grad*/
// Topo Sort
void __topoSort(std::shared_ptr<valueData> &node,
                std::vector<std::shared_ptr<valueData>> &order) {

	for (auto &child : node->children) {
		__topoSort(child, order);
	}
	order.push_back(node);
}

// Backward
void Value::backward() {

	if (this->ptr->total_size != 1) {
		std::exit(EXIT_FAILURE);
	}
	std::vector<std::shared_ptr<valueData>> order;
	__topoSort(this->ptr, order);
	std::reverse(order.begin(), order.end());
	*(this->ptr->grad) = 1.0;
	for (auto node : order) {
		if (node->_backward != nullptr) {
			node->_backward();
		}
	}
}

