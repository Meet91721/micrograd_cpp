#include "Value.h"

int main()
{
	Value a = Value({1, 2, 3}, "a");
	a = T<3>{
		{
			{
				{1, 2, 3}
			},
			{
				{5, 12, 23}
			}
		}
	};
	std::cout << "This one here: " << a({0,0,0}) << '\n';
	a({0,0,0}) = T<3>{
		{
			{
				312
			}
		}
	};
	std::cout << a;
}
