#pragma once
#include <_strings.h>
#include <initializer_list>
#include <memory>
#include <vector>
#include <functional>
#include <iostream>
#include <numeric>
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

struct Selector{
	int *indexes;
	int n_elements;
	bool isRange;
	Selector(int index){
		indexes = new int[1];
		indexes[0] = index;
		n_elements = 1;
		isRange = false;
	}
	Selector(std::initializer_list<int> range){
		assert(range.size() == 2 && "Range only accepts the first and last element");
		indexes = new int[2];
		std::copy(range.begin(), range.end(), indexes);
		n_elements = indexes[1] - indexes[0];
		isRange = true;
	}
	Selector(int n, std::initializer_list<int> elems){
		indexes = new int[n];
		std::copy(elems.begin(), elems.end(), indexes);
		n_elements = n;
		isRange = false;
	}
	Selector(Selector &&sel) noexcept {
		indexes = sel.indexes;
		isRange = sel.isRange;
		n_elements = sel.n_elements;
		sel.indexes = nullptr;
	}
	Selector& operator()(Selector &&other) noexcept {
		delete[] indexes;
		indexes = other.indexes;
		isRange = other.isRange;
		n_elements = other.n_elements;
		other.indexes = nullptr;
		return *this;
	}
	Selector(const Selector &sel) noexcept : Selector(std::move(const_cast<Selector&>(sel))){
	}
	Selector& operator=(const Selector &other) noexcept {
		return *this = std::move(const_cast<Selector&>(other));
	}
	void isValid(int mx) const{
		if(n_elements == 1){
			assert(*indexes < mx && "Tried accessing out of index element");
		}
		else if(isRange == true){
			assert(indexes[0] < indexes[1] && "Start element must be lesser than the end index");
			assert(indexes[1] < mx && "Tried accessing out of index element");
		}
		else{
			for(int i = 0; i < n_elements; i++){
				assert(indexes[i] < mx && "Tried accessing out of index element");
			}
		}
	}
	~Selector(){
		delete[] indexes;
	}
};

class Projection{
public:
	double *t;
	int n_elements;
	std::vector<int> shape;

	void calculate_size(std::vector<Selector> &indices){
		for(int i = 0; i < indices.size(); i++){
			if(indices[i].isRange){
				int elements = indices[i].indexes[1] - indices[i].indexes[0];
				n_elements *= elements;
				shape.push_back(elements);
			}else{
				int elements = n_elements;
				n_elements += indices[i].n_elements;
				shape.push_back(elements);
			}
		}
	}

	Projection(std::vector<Selector> &indices, double *ptr){
		calculate_size(indices);
		t = new double[n_elements];
		// delete this in future
		for(int i = 0; i < n_elements; i++){
			t[i] = i;
		}
	}
	
	void printer(){
		for(auto it: shape){
			std::cout << it << " ";
		}
		std::cout << '\n';
		for(int i = 0; i < n_elements; i++){
			std::cout << t[i] << " ";
		}
	}

	~Projection(){
		delete t;
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

	Value operator()(std::vector<Selector> &&indexes);

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


