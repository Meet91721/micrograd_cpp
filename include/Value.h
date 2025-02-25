#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <cassert>

class Value;

class valueData{

public:
	double *data;
	int total_size;
	std::vector<int> shape;
	std::string label;
	std::vector<std::shared_ptr<valueData>> children;
	double *grad;
	std::function<void()>_backward;
	std::string op;

	/* explicitly deleting the default constructor */
	valueData() = delete;

	/* main constructor */
	valueData(double _data,std::string _label = "",std::vector<std::shared_ptr<valueData>> _children = {},std::string _op = "",std::function<void()> __backward = nullptr):label(std::move(_label)),children(std::move(_children)),op(std::move(_op)),_backward(__backward) {
		total_size = 1;
		data = new double(_data);
		grad = new double(0);
		shape = {};
	}

	valueData(std::vector<int> _shape,std::string _label = "",std::vector<std::shared_ptr<valueData>> _children = {},std::string _op = "",std::function<void()> __backward = nullptr):label(std::move(_label)),children(std::move(_children)),op(std::move(_op)),_backward(__backward) {
		total_size = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<int>());
		data = new double[total_size];
		grad = new double[total_size];
		shape = _shape;
	}

	/* destructor */
	~valueData(){
		delete data;
		delete grad;
	}
};

class Value{
private:
	inline static std::vector<std::shared_ptr<valueData>> tmpObjs;

public:
	std::shared_ptr<valueData> ptr;

	/* explicitly deleting the default constructor */
	Value() = delete;

	/* main constructor */
	Value(double _data,std::string _label = "",std::vector<std::shared_ptr<valueData>> _children = {},std::string _op = "",std::function<void()> __backward = nullptr){
		ptr = std::make_shared<valueData>(_data,_label,_children,_op,__backward);
	}

	Value(std::vector<int> _shape,std::string _label = "",std::vector<std::shared_ptr<valueData>> _children = {},std::string _op = "",std::function<void()> __backward = nullptr){
		ptr = std::make_shared<valueData>(_shape,_label,_children,_op,__backward);
	}

	/* move constructor */
	Value(Value&& other){
		this->ptr = other.ptr;
		other.ptr.reset();
	}

	// copy constructor
	Value(const Value& other){
		// Making changes following pytorch's psychology
		this->ptr = other.ptr;
	}

	/* assignment copy constructor */
	Value& operator=(const Value& other){
		if(this != &other){
			if(this->ptr != nullptr){
				Value::tmpObjs.push_back(this->ptr);
			}
			this->ptr = other.ptr;
		}
		return *this;
	}

	/* assignment move constructor */
	Value& operator=(Value&& other) noexcept {
		if(this != &other){
			if(this->ptr != nullptr){
				Value::tmpObjs.push_back(this->ptr);
			}
			this->ptr = other.ptr;
			other.ptr.reset();
		}
		return *this;
	}

	Value operator=(double data) {
		assert(this->ptr->shape.size() == 0 && "Shape conflict");
		*(this->ptr->data) = data;
		return *this;
	}

	/* destructor */
	~Value(){}

	/* getter functions */
	double* getdata(){
		return this->ptr->data;
	}

	double* getGrad(){
		return this->ptr->grad;
	}

	/* for printing the Value object (right now implemented by overloading the ostream << operator)*/
	friend std::ostream& operator<<(std::ostream&, Value&) noexcept;

	/* for printing the Value object (right now implemented by overloading the ostream << operator)*/
	friend std::ostream& operator<<(std::ostream&, Value&&) noexcept;

	/* backward function */
	void backward();

	// addition
	Value operator+(Value&);
	Value operator+(double);
	friend Value operator+(double, Value&);

	// multiplication
	Value operator*(Value&);
	Value operator*(double);
	friend Value operator*(double, Value&);

	// substraction
	Value operator-(Value&);
	Value operator-(double);
	friend Value operator-(double, Value&);

	// division
	Value operator/(Value&);
	Value operator/(double);
	friend Value operator/(double, Value&);

	//power
	Value operator^(Value&);
	Value operator^(double);
	friend Value operator^(double, Value&);
};

