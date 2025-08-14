#include <catch2/catch_test_macros.hpp>

#include <map>
#include <string>
#include <vector>

using namespace std;

map<string, int> table; 

int cube(int x) { return x * x * x;}

string concatinate(string a, string b) 
{
    return a + b;
}

string concatinate(vector<string> vec) 
{
    string str = "";
    for (int c = 0; c < vec.size(); c++) 
    {
        str.append(vec[c]);
    }
    return str;
}

int get_value(string key, int value) 
{
    table[key] = value; 
    return table[key];  
}

TEST_CASE("concatenate function", "[concatinate]") 
{
    SECTION("concatenate two strings") 
    {
        REQUIRE(concatinate("Hello, ", "World!") == "Hello, World!");
    }

    SECTION("concatenate vector of strings") 
    {
        vector<string> vec = {"Hello, ", "World", "!"};
        REQUIRE(concatinate(vec) == "Hello, World!");
    }
}

TEST_CASE("get value function", "[get value]")
{
    REQUIRE(get_value("test", 42) == 42);
}

TEST_CASE("cube function", "[cube]") 
{
    SECTION("cube of a positive number") 
    {
        REQUIRE(cube(3) == 27);
    }

    SECTION("cube of zero") 
    {
        REQUIRE(cube(0) == 0);
    }

    SECTION("cube of a negative number") 
    {
        REQUIRE(cube(-2) == -8);
    }
}