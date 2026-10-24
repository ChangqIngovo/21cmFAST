// Independent priority-queue oracle. The actual production queue is included.
#include "calendar.cpp"
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

static void exercise(double duration) {
    constexpr int count=5000;
    Heap queue(count,duration);
    std::set<std::pair<double,int>> oracle;
    std::vector<double> times(count,0);
    std::vector<bool> active(count,false);
    auto check=[&]() {
        if(queue.empty()!=oracle.empty())throw std::runtime_error("empty mismatch");
        if(!oracle.empty()) {
            auto actual=queue.top();auto expected=*oracle.begin();
            if(actual.time!=expected.first||actual.node!=expected.second)
                throw std::runtime_error("priority mismatch");
        }
    };
    auto update=[&](int node,double time) {
        if(active[node])oracle.erase({times[node],node});
        times[node]=time;active[node]=true;oracle.insert({time,node});
        queue.update(node,time);check();
    };
    auto erase=[&](int node) {
        if(active[node])oracle.erase({times[node],node});
        active[node]=false;queue.erase(node);check();
    };
    auto pop=[&]() {
        if(oracle.empty())return;
        auto expected=*oracle.begin();oracle.erase(oracle.begin());active[expected.second]=false;
        queue.pop();check();
    };
    // Identical timestamps force the bounded-size bucket's indexed-heap path.
    for(int node=count-1;node>=0;node--)update(node,duration*.5);
    for(int node=0;node<count;node+=3)update(node,std::nextafter(duration*.5,0.));
    for(int node=1;node<count;node+=3)update(node,std::nextafter(duration*.5,duration));
    while(!oracle.empty())pop();
    // Tiny values, exact bin edges, and each immediately adjacent double.
    std::vector<double> special={0.,std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::min(),std::nextafter(duration,0.)};
    for(int k=0;k<=PC_CALENDAR_BUCKETS;k+=std::max(1,PC_CALENDAR_BUCKETS/16)) {
        double edge=duration*(double(k)/PC_CALENDAR_BUCKETS);
        special.push_back(edge);special.push_back(std::nextafter(edge,0.));
        special.push_back(std::nextafter(edge,duration));
    }
    std::mt19937_64 random(295482);
    for(int step=0;step<100000;step++) {
        int node=random()%count,operation=random()%10;
        if(operation<6) {
            double time=step%3?duration*std::generate_canonical<double,53>(random):special[random()%special.size()];
            update(node,time);
        } else if(operation<8)erase(node);else pop();
    }
    while(!oracle.empty())pop();
    // Dynamic node reservation preserves membership and ordering.
    queue.reserve_nodes(2*count);queue.update(2*count-1,0.);
    if(queue.top().node!=2*count-1)throw std::runtime_error("resize failure");
    queue.pop();if(!queue.empty())throw std::runtime_error("final empty failure");
}
int main() {
    exercise(1.);exercise(1e-300);exercise(1e300);
    std::cout<<"calendar priority oracle passed, buckets="<<PC_CALENDAR_BUCKETS<<"\n";
}
