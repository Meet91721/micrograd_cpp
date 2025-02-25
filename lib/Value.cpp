#include "Value.h"
#include <algorithm>
#include <cmath>
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

