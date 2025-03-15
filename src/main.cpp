#include "Value.h"

int main()
{
	// T<3> v = {{{1, 2}, {3, 4}}};
	// Value a = Value(v);
	// Value a = Value(T<2> {
	// 	{1, 2, 3},
	// 	{6, 5, 4}
	// });
	Value a = Value(2);
	Value b = Value(33);


	// Value a = Value(21);
	std::cout << a;
	std::cout << b;
	Value c = a * b;
	c.backward();
	std::cout << *(a.ptr->grad) << '\n';
	std::cout << *(b.ptr->grad) << '\n';
	std::cout << *(c.ptr->grad) << '\n';
}
