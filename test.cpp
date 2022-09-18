#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

inline double drand(void)
{
    // RAND_MAX + 1 to avoid drand() == 1.0
    return (double) rand() / ((double) RAND_MAX + 1);
}

// Return random int i, 0 <= i < limit

inline int irand(int limit)
{
    return int(limit * drand());
}

void crossover(vector<int> &a, vector<int> &b, vector<int> &c, vector<int> &d)
{
    // assert(a.size() == b.size());

    const int n = a.size();

    // static to avoid realloc every call
    std::vector<int> lookup_c(n);
    std::vector<int> lookup_d(n);
    c.resize(n);
    d.resize(n);

    // Create lookup tables and initialize c and d
    for (int i = 0; i != n; i++)
    {
        c[i] = a[i];
        lookup_c[c[i]] = i;

        d[i] = b[i];
        lookup_d[d[i]] = i;
    }

    // Crossover random sequence of at least n / 4 and at most 3/4 * n cities
    const int start = 3;
    const int end = 6;

    // Do the PMX
    for (int j = start; j != end; j++)
    {
        // const int j = i % n; // i mod n

        const int posb = lookup_c[b[j]];
        std::swap(c[posb], c[j]);
        lookup_c[c[posb]] = posb;

        const int posa = lookup_d[a[j]];
        std::swap(d[posa], d[j]);
        lookup_d[d[posa]] = posa;
    }
}

int main(){
    vector<int> a = {0,1,2,3,4,5,6,7,8}, b = {3,7,5,1,6,8,2,4,0}, c, d;
    crossover(a,b,c,d);
    for_each(c.begin(), c.end(), [](int item){cout<<item<<" ";});
    cout<<endl;
    for_each(d.begin(), d.end(), [](int item){cout<<item<<" ";});
}
