#include "Value.h"
#include <gtest/gtest.h>

namespace {

TEST(addTest, basicTests) {

  GTEST_LOG_(INFO) << "add test started\n";

  Value a(2);
  Value b(3);
  Value c = std::move(a + b);

  /* EXPECT_EQ(c.data,5.0);
  EXPECT_EQ(c.grad, 0.0);

  c.grad = 2.0;
  c._backward(c);

  EXPECT_EQ(a.grad, 2.0);
  EXPECT_EQ(a.grad, 2.0);

  Value d = a + 2.0;
  EXPECT_EQ(d.data, 4.0);

  d.grad = 4.0;
  d._backward(d);

  EXPECT_EQ(a.grad, 6.0);

  Value e = 10.0 + b;

  e.grad = 15.0;
  e._backward(e);

  EXPECT_EQ(b.grad, 17.0); */

  GTEST_LOG_(INFO) << "add tests completed\n";
}

} // namespace
