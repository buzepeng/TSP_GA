#include <vector>
#include <iostream>
#include <sstream>
#include <tuple>

using namespace std;


ostream& operator << (ostream& output, tuple<vector<int>, int, int> t)
{
    vector<int> a = std::get<0>(t);
    int start = std::get<1>(t), end = std::get<2>(t);
    for(int i = start;i<end;i++)    output<<a[i]<<" ";
    return output;
}

int main(){
    vector<int> a(10, 0);
    cout<<std::make_tuple(a, 0, 10);
}