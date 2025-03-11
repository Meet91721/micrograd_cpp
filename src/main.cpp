#include "Value.h"

int main()
{
	Value a = Value({1, 2, 3}, "a");
	a = std::vector<std::vector<std::vector<double>>> {
		{
			{
				{1, 2, 3}
			},
			{
				{5, 12, 23}
			}
		}
	};
	std::cout << a;
}
