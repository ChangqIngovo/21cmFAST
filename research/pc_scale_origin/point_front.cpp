/* Dense independent-grid reference for static point-source opaque-cell fronts.
 * Sources have explicit positions. All directions are emitted simultaneously;
 * only exhaustion of gas cells advances the shared normalized emission clock.
 * This research reference intentionally materializes every source ray at t=0.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using Real = long double;
struct Ray { Real travelled=0, arrival=0, rate=0; int voxel[3], source, direction, next=-1; };
struct Event {Real time; int cell;};
class Heap {
    std::vector<Event> data; std::vector<int> position;
    static bool before(const Event&a,const Event&b) {return a.time<b.time||(a.time==b.time&&a.cell<b.cell);}
    void swap(int a,int b){std::swap(data[a],data[b]);position[data[a].cell]=a;position[data[b].cell]=b;}
    void up(int at){while(at){int p=(at-1)/2;if(!before(data[at],data[p]))break;swap(at,p);at=p;}}
    void down(int at){while(2*at+1<(int)data.size()){int k=2*at+1;if(k+1<(int)data.size()&&before(data[k+1],data[k]))k++;
        if(!before(data[k],data[at]))break;swap(k,at);at=k;}}
public:
    explicit Heap(int size):position(size,-1){}
    bool empty()const{return data.empty();}
    Event pop(){Event out=data[0];position[out.cell]=-1;if(data.size()==1)data.pop_back();else{data[0]=data.back();data.pop_back();position[data[0].cell]=0;down(0);}return out;}
    void update(int cell,Real time){int at=position[cell];if(at<0){position[cell]=data.size();data.push_back({time,cell});up(data.size()-1);}else{
        Real old=data[at].time;data[at].time=time;if(time<old)up(at);else down(at);}}
};

extern "C" int point_evolve(int n,int ns,int nk,const double*positions,const double*photons,
                            const double*initial,const double*directions,double horizon,int periodic,
                            double*absorbed,double*source_absorbed,double*escaped,double*retained,
                            double*stats) {
    if(n<2||ns<0||nk<1||!std::isfinite(horizon)||horizon<=0)return 1;
    const int64_t count64=(int64_t)n*n*n, rays64=(int64_t)ns*nk;
    if(count64>std::numeric_limits<int>::max()||rays64>std::numeric_limits<int>::max())return 2;
    const int count=(int)count64;
    std::vector<Real> remaining(initial,initial+count),rate(count,0),last(count,0),used(count,0);
    std::vector<Real> source_used(ns,0),source_escape(ns,0),source_retain(ns,0);
    std::vector<int> head(count,-1);
    std::vector<Ray> rays; rays.reserve(rays64);
    Heap heap(count);
    int64_t queries=0,events=0,assignments=0;
    auto consume=[&](int cell,Real now){
        if(now<last[cell])return false;
        Real amount=std::min(remaining[cell],rate[cell]*(now-last[cell]));
        used[cell]+=amount;remaining[cell]-=amount;last[cell]=now;return true;
    };
    auto attach=[&](int ri,Real now)->int {
        Ray &r=rays[ri];const double*v=directions+3*r.direction;const double*origin=positions+3*r.source;
        for(int64_t steps=0;steps<10000000;steps++){
            queries++;
            bool outside=false;int ijk[3];
            for(int a=0;a<3;a++){outside|=r.voxel[a]<0||r.voxel[a]>=n;ijk[a]=r.voxel[a]%n;if(ijk[a]<0)ijk[a]+=n;}
            if(outside&&!periodic){source_escape[r.source]+=r.rate*(1-now);return 0;}
            if(r.travelled>=(Real)horizon){source_retain[r.source]+=r.rate*(1-now);return 0;}
            int cell=(ijk[0]*n+ijk[1])*n+ijk[2];
            if(remaining[cell]>0){
                if(!consume(cell,now))return 3;
                rate[cell]+=r.rate;r.arrival=now;r.next=head[cell];head[cell]=ri;assignments++;
                Real end=remaining[cell]>0?now+remaining[cell]/rate[cell]:now;
                heap.update(cell,end);return 0;
            }
            Real crossing[3],next=std::numeric_limits<Real>::infinity();
            for(int a=0;a<3;a++){
                crossing[a]=std::numeric_limits<Real>::infinity();
                if(v[a]!=0){Real face=(Real)r.voxel[a]+(v[a]>0?1:0);crossing[a]=(face-(Real)origin[a])/(Real)v[a];next=std::min(next,crossing[a]);}
            }
            if(!std::isfinite(next)||next<r.travelled)return 4;
            if(next>(Real)horizon){source_retain[r.source]+=r.rate*(1-now);return 0;}
            for(int a=0;a<3;a++)if(crossing[a]==next)r.voxel[a]+=v[a]>0?1:-1;
            r.travelled=next;
        }
        return 5;
    };
    for(int s=0;s<ns;s++){
        if(photons[s]<=0)continue;
        for(int k=0;k<nk;k++){
            Ray r;r.source=s;r.direction=k;r.rate=(Real)photons[s]/nk;
            for(int a=0;a<3;a++){
                const double x=positions[3*s+a];r.voxel[a]=(int)std::floor(x);
                if(directions[3*k+a]<0&&x==std::floor(x))r.voxel[a]--;
            }
            rays.push_back(r);int rc=attach((int)rays.size()-1,0);if(rc)return rc;
        }
    }
    while(!heap.empty()){
        Event e=heap.pop();if(e.time>=1)break;
        int cell=e.cell;Real now=e.time;
        if(remaining[cell]==0&&head[cell]<0)continue;
        // The heap records the shared-cell exhaustion time, so all its rays
        // depart together. Force the endpoint rather than leaving roundoff gas.
        used[cell]=(Real)initial[cell];remaining[cell]=0;last[cell]=now;events++;
        int ri=head[cell];head[cell]=-1;rate[cell]=0;
        while(ri>=0){int next=rays[ri].next;
            source_used[rays[ri].source]+=rays[ri].rate*(now-rays[ri].arrival);
            int rc=attach(ri,now);if(rc)return rc;ri=next;
        }
    }
    for(int cell=0;cell<count;cell++){
        if(!consume(cell,1))return 3;
        absorbed[cell]=(double)used[cell];
        for(int ri=head[cell];ri>=0;ri=rays[ri].next)
            source_used[rays[ri].source]+=rays[ri].rate*(1-rays[ri].arrival);
    }
    for(int s=0;s<ns;s++){source_absorbed[s]=(double)source_used[s];escaped[s]=(double)source_escape[s];retained[s]=(double)source_retain[s];}
    stats[0]=events;stats[1]=queries;stats[2]=assignments;stats[3]=rays.size();stats[4]=sizeof(Ray);stats[5]=rays.capacity()*sizeof(Ray);
    return 0;
}
