#include <atomic>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
  std::atomic<int> value{0};
  std::thread worker([&]() { value.store(7); });
  worker.join();
  const std::vector<float> values{3.f, 4.f};
  const float magnitude = std::sqrt(values[0]*values[0] + values[1]*values[1]);
  std::printf("ARMv6 C++ smoke: atomic=%d magnitude=%.1f\n", value.load(), magnitude);
  return value.load() == 7 && magnitude == 5.f ? 0 : 1;
}
