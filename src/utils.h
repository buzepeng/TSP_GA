bool isStringNumber(std::string str){
    if(str.size()==0)   return false;
    for(int i=0;i<str.length();i++){
        if(str[i]<'0'||str[i]>'9'){
        return false;
        }
    }
    return true; 
}

int getCeilPowerOfTwo(int num)
{
	--num;
	num |= num >> 1;
	num |= num >> 2;
	num |= num >> 4;
	num |= num >> 8;
	num |= num >> 16;
	return ++num;
}

#define START_TEST std::cout<<"test"<<std::endl;
#define END_TEST std::cout<<"test complete"<<std::endl;

template<typename T>
std::ostream& operator << (std::ostream& output, std::tuple<thrust::device_vector<T>, int, int> t)
{
    thrust::host_vector<T> a = std::get<0>(t);
    int row = std::get<1>(t), col = std::get<2>(t);
    for(int i = 0;i<row;i++){
        for(int j = 0;j<col;j++){
            output<<a[i*col+j]<<' ';
        }
        output<<std::endl;
    }
    return output;
}

void check_pop(thrust::device_vector<int> &d_pop, int s, int N){
    bool illegal = false;
    thrust::host_vector<int> pop = d_pop;
    for(int i = 0;i<s;i++){
        thrust::host_vector<int> t(pop.begin()+i*N, pop.begin()+(i+1)*N);
        thrust::sort(thrust::host, t.begin(), t.end());
        for(int j = 0;j<N;j++){
            if(t[j]!=j){
                illegal = true;
                std::cout<<"pop "<<i<<" pos "<<j<<" incorrect"<<std::endl;
                break;
            }
        }
        if(illegal){
            std::cout<<"illegal pop"<<std::endl;
            for(int j = 0;j<N;j++){
                std::cout<<t[j]<<" ";
            }
            std::cout<<std::endl;
            return;
        }
    }
    std::cout<<"legal pop"<<std::endl;
}