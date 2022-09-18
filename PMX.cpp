#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <iostream>

using namespace std;

inline int	  RandInt(int x,int y) {return rand()%(y-x+1)+x;}

//returns a random float between zero and 1
inline double RandFloat()		   {return (rand())/(RAND_MAX+1.0);}

// void CrossoverPMX(const vector<int>	&mum, const vector<int>	&dad, vector<int> &baby1, vector<int> &baby2)
// {
// 	baby1 = mum;
// 	baby2 = dad;
	
// 	//just return dependent on the crossover rate or if the
// 	//chromosomes are the same.
// 	// if ( (RandFloat() > m_dCrossoverRate) || (mum == dad)) 
// 	// {
// 	// 	return;
// 	// }

// 	//first we choose a section of the chromosome
// 	int beg = RandInt(0, mum.size()-2);
	
// 	int end = beg;
	
// 	//find an end
// 	while (end <= beg)
// 	{
// 		end = RandInt(0, mum.size()-1);
// 	}

// 	//now we iterate through the matched pairs of genes from beg
// 	//to end swapping the places in each child
// 	vector<int>::iterator posGene1, posGene2;

// 	for (int pos = beg; pos < end+1; ++pos)
// 	{
// 		//these are the genes we want to swap
// 		int gene1 = mum[pos];
// 		int gene2 = dad[pos];

// 		if (gene1 != gene2)
// 		{
// 			//find and swap them in baby1
// 			posGene1 = find(baby1.begin(), baby1.end(), gene1);
// 			posGene2 = find(baby1.begin(), baby1.end(), gene2);

// 			swap(*posGene1, *posGene2);

// 			//and in baby2
// 			posGene1 = find(baby2.begin(), baby2.end(), gene1);
// 			posGene2 = find(baby2.begin(), baby2.end(), gene2);
			
// 			swap(*posGene1, *posGene2);
// 		}
		
// 	}//next pair
// }
void swap(int *a, int *b){
	int temp = *a;
	*a = *b;
	*b = temp;
}

void CrossoverPMX(vector<int> mum, vector<int> dad, vector<int> baby1, vector<int> baby2){
	baby1 = mum;
	baby2 = dad;
	int n = mum.size();
	vector<int> lookup1(n), lookup2(n);
	for(int i = 0;i<n;i++){
		lookup1[baby1[i]] = i;
		lookup2[baby2[i]] = i;
	}
	int start = 3, end = 7;
	for(int j = start;j<=end;j++){
		int gene1 = mum[j], gene2 = dad[j], posgene1, posgene2;
		posgene1 = lookup1[gene1];
		posgene2 = lookup1[gene2];
		swap(&baby1[posgene1], &baby1[posgene2]);
		swap(&lookup1[baby1[posgene1]], &lookup1[baby1[posgene2]]);

		posgene1 = lookup2[gene1];
		posgene2 = lookup2[gene2];
		swap(&baby2[posgene1], &baby2[posgene2]);
		swap(&lookup2[baby2[posgene1]], &lookup2[baby2[posgene2]]);
	}
}

int main(){
    vector<int> a = {0,1,2,3,4,5,6,7,8}, b = {3,7,5,1,6,8,2,4,0}, c(9), d(9);
    CrossoverPMX(a,b,c,d);
    for_each(c.begin(), c.end(), [](int item){cout<<item<<" ";});
    cout<<endl;
    for_each(d.begin(), d.end(), [](int item){cout<<item<<" ";});
}