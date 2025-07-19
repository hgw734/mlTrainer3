#include <iostream>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> items = {"C++", "standard", "library", "working"};
    
    for (const auto& item : items) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    return 0;
}