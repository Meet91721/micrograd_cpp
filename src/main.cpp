#include "Value.h"

int main()
{
	Value a(2,"a");
	Value b(3,"b");
	Value c = a-b;
	Value d = 2 ^ c;
	d.backward();
	std::cout << a << '\n';
	std::cout << b << '\n';
	std::cout << c << '\n';
	std::cout << d;
}
