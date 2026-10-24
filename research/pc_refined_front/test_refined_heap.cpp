// Exercise the actual transport heap against an independent ordered reference.
#include "refined.cpp"
#include <map>
#include <random>
#include <iostream>
int main(){
    {
        Heap h;h.ensure(48);
        std::vector<Real> v(48,1000);
        v[0]=0;v[1]=100;v[2]=1;v[3]=2;v[4]=3;
        v[5]=200;v[6]=210;v[7]=220;v[8]=230;v[9]=10;
        v[33]=250;v[34]=249;v[35]=248;v[36]=247;v[37]=20;
        for(int i=0;i<48;i++)h.update(i,v[i]);
        v[8]=300;h.update(8,v[8]);
        for(int step=0;step<48;step++){
            int best=std::min_element(v.begin(),v.end())-v.begin();
            Event e=h.top();
            if(e.node!=best||e.time!=v[best]){std::cerr<<"crafted heap mismatch at "<<step<<"\n";return 4;}
            h.erase(e.node);v[e.node]=std::numeric_limits<Real>::infinity();
        }
    }
    constexpr int n=200;
    Heap heap; heap.ensure(n);
    std::map<int,Real> values;
    std::mt19937 random(19642);
    for(int step=0;step<100000;step++){
        int id=random()%n;
        if(step%257==0) {
            while(!values.empty()) {
                auto best=std::min_element(values.begin(),values.end(),[](auto&a,auto&b){return a.second<b.second||(a.second==b.second&&a.first<b.first);});
                Event e=heap.top();
                if(e.node!=best->first||e.time!=best->second){std::cerr<<"drain mismatch at "<<step<<"\n";return 3;}
                heap.erase(e.node);values.erase(e.node);
            }
        }
        if(random()%4==0){heap.erase(id);values.erase(id);}
        else {Real value=(random()%100000)/Real(32);heap.update(id,value);values[id]=value;}
        if(heap.empty()!=values.empty())return 1;
        if(!values.empty()){
            auto best=std::min_element(values.begin(),values.end(),[](auto&a,auto&b){return a.second<b.second||(a.second==b.second&&a.first<b.first);});
            Event e=heap.top();
            if(e.node!=best->first||e.time!=best->second){std::cerr<<"heap mismatch at step "<<step<<"\n";return 2;}
        }
    }
    return 0;
}
