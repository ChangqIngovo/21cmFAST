/* Research event integrator for constant cell-centred point sources,
 * a sharp opaque voxel front, and infinite propagation speed. No chemistry.
 * Only exhausted cells generate events. Rays keep directions and positions.
 */
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

struct Ray {
    double x[3], rate, arrival, travelled;
    int source, direction, next;
};

// Fixed blocks retain stable ray indices without vector growth copying all
// materialized rays or temporarily holding both old and doubled allocations.
class RayStore {
    static constexpr size_t block_size=4096;
    std::vector<std::unique_ptr<Ray[]>> blocks;
    size_t count=0;
public:
    size_t size() const {return count;}
    size_t allocated_bytes() const {return blocks.size()*block_size*sizeof(Ray);}
    Ray &operator[](size_t at) {return blocks[at/block_size][at%block_size];}
    void push_back(const Ray &ray) {
        if(count%block_size==0)blocks.emplace_back(new Ray[block_size]);
        (*this)[count]=ray;count++;
    }
};
struct Event {
    double time;
    int cell;
};

// Exactly one scheduled exhaustion per opaque receiving cell. Indexed updates
// avoid a stale event allocation for every newly arriving ray.
class EventHeap {
    std::vector<Event> data;
    std::vector<int> position;
    static bool earlier(const Event &a, const Event &b) {
        return a.time<b.time || (a.time==b.time && a.cell<b.cell);
    }
    void exchange(int a, int b) {
        std::swap(data[a],data[b]);
        position[data[a].cell]=a;position[data[b].cell]=b;
    }
    void up(int at) {
        while(at>0) {
            int parent=(at-1)/2;
            if(!earlier(data[at],data[parent]))break;
            exchange(at,parent);at=parent;
        }
    }
    void down(int at) {
        const int size=(int)data.size();
        while(2*at+1<size) {
            int child=2*at+1;
            if(child+1<size&&earlier(data[child+1],data[child]))child++;
            if(!earlier(data[child],data[at]))break;
            exchange(at,child);at=child;
        }
    }
public:
    int64_t updates=0;
    size_t peak=0;
    explicit EventHeap(int cells):position(cells,-1) {}
    bool empty() const {return data.empty();}
    const Event &top() const {return data.front();}
    void update(int cell,double time) {
        updates++;
        int at=position[cell];
        if(at<0) {
            if(data.size()==data.capacity()) {
                size_t grown=std::max((size_t)16,2*data.capacity());
                data.reserve(std::min(position.size(),grown));
            }
            position[cell]=(int)data.size();
            data.push_back({time,cell});
            peak=std::max(peak,data.size());
            assert(data.size()<=position.size());
            up((int)data.size()-1);
        } else {
            double previous=data[at].time;
            data[at].time=time;
            // Rates only increase, but allow one-ulp upward changes caused by
            // lazy capacity subtraction, preserving the old queue semantics.
            if(time<previous)up(at);else down(at);
        }
    }
    void pop() {
        position[data.front().cell]=-1;
        if(data.size()==1) {data.pop_back();return;}
        data.front()=data.back();data.pop_back();
        position[data.front().cell]=0;down(0);
    }
};

static thread_local int64_t last_heap_peak=0,last_heap_updates=0,last_ray_bytes=0;
extern "C" int64_t event_front_last_max_heap_nodes() {return last_heap_peak;}
extern "C" int64_t event_front_last_heap_updates() {return last_heap_updates;}
extern "C" int64_t event_front_last_ray_allocated_bytes() {return last_ray_bytes;}

extern "C" int evolve_events(int n, const double *source, const double *initial_capacity,
                             const int32_t *clearance, int nd, const double *directions,
                             double duration, double max_distance, int periodic,
                             double *absorbed, double *source_absorbed,
                             double *source_escaped, double *source_retained,
                             double *angle_absorbed, double *angle_escaped,
                             double *angle_retained, double *statistics) {
    if(n<1||nd<1||duration<=0||max_distance<=0||
       !std::isfinite(duration)||!std::isfinite(max_distance)) return 1;
    if((int64_t)n*n>INT_MAX/n)return 5;
    const int cells=n*n*n;
    std::vector<double> capacity(initial_capacity,initial_capacity+cells);
    std::vector<double> rate(source,source+cells),last(cells,0);
    std::vector<int> head(cells,-1);
    std::vector<uint8_t> emitted(cells,0);
    RayStore rays;
    EventHeap heap(cells);
    double local_absorbed=0;
    int64_t events=0,queries=0,assignments=0,stale=0;
    auto schedule = [&](int cell,double now) {
        if(capacity[cell]>0 && rate[cell]>0)
            heap.update(cell,now+capacity[cell]/rate[cell]);
    };
    for(int i=0;i<cells;i++) schedule(i,0);

    // Associate a ray with its first still-opaque cell at this event time.
    auto advance = [&](int ri,double now) -> int {
        Ray &ray=rays[ri];
        const double *v=directions+3*ray.direction;
        int answer=-1;
        for(int count=0;count<1000000;count++) {
            queries++;
            bool outside=false;
            for(int a=0;a<3;a++) {
                if(ray.x[a]<0 || ray.x[a]>=n) {
                    if(periodic) {
                        ray.x[a]=std::fmod(ray.x[a],(double)n);
                        if(ray.x[a]<0)ray.x[a]+=n;
                    } else outside=true;
                }
            }
            if(outside) {answer=-1;break;}
            int ijk[3]={(int)std::floor(ray.x[0]),(int)std::floor(ray.x[1]),(int)std::floor(ray.x[2])};
            int cell=(ijk[0]*n+ijk[1])*n+ijk[2];
            if(capacity[cell]>0) {answer=cell;break;}
            double step=std::numeric_limits<double>::infinity();
            if(clearance[cell]>=2) {
                step=clearance[cell]-1.0000000001;
                // This clearance is open-box and hence also conservative at a wrap.
            } else {
                for(int a=0;a<3;a++) {
                    if(v[a]==0)continue;
                    double face=v[a]>0?ijk[a]+1.0:ijk[a];
                    step=std::min(step,(face-ray.x[a])/v[a]);
                }
                step+=1e-10;
            }
            if(!std::isfinite(step)||step<=0)return 2;
            if(ray.travelled+step>max_distance) {answer=-2;break;}
            for(int a=0;a<3;a++)ray.x[a]+=step*v[a];
            ray.travelled+=step;
            if(count==999999)return 3;
        }
        if(answer<0) {
            double future=ray.rate*(duration-now);
            if(answer==-1) {
                source_escaped[ray.source]+=future;
                angle_escaped[ray.direction]+=future;
            } else {
                source_retained[ray.source]+=future;
                angle_retained[ray.direction]+=future;
            }
            ray.next=-1;return 0;
        }
        // Lazy update of this cell's old total incident rate before it changes.
        capacity[answer]-=rate[answer]*(now-last[answer]);
        if(capacity[answer]<0)capacity[answer]=0;
        last[answer]=now;
        rate[answer]+=ray.rate;
        ray.arrival=now;
        ray.next=head[answer];head[answer]=ri;
        assignments++;
        if(capacity[answer]<=0) {
            // Exact simultaneous exhaustion: still process its incoming rays.
            heap.update(answer,now);
        } else schedule(answer,now);
        return 0;
    };
    auto emit = [&](int cell,double now) -> int {
        emitted[cell]=1;
        local_absorbed+=source[cell]*now;
        source_absorbed[cell]+=source[cell]*now;
        int ix=cell/(n*n),iy=(cell/n)%n,iz=cell%n;
        for(int k=0;k<nd;k++) {
            if(rays.size()>=(size_t)INT_MAX)return 4;
            rays.push_back({{ix+0.5,iy+0.5,iz+0.5},source[cell]/nd,now,0,cell,k,-1});
            int rc=advance((int)rays.size()-1,now);if(rc)return rc;
        }
        return 0;
    };
    for(int cell=0;cell<cells;cell++) {
        if(capacity[cell]<=0&&source[cell]>0) {
            int rc=emit(cell,0);if(rc)return rc;
        }
    }
    while(!heap.empty()) {
        Event event=heap.top();heap.pop();
        if(event.time>=duration)break;
        const int cell=event.cell;const double now=event.time;
        capacity[cell]=0;last[cell]=now;events++;
        // This cell never becomes opaque again in the declared model.
        int ri=head[cell];head[cell]=-1;
        while(ri>=0) {
            int next=rays[ri].next;
            double amount=rays[ri].rate*(now-rays[ri].arrival);
            source_absorbed[rays[ri].source]+=amount;
            angle_absorbed[rays[ri].direction]+=amount;
            int rc=advance(ri,now);if(rc)return rc;
            ri=next;
        }
        if(source[cell]>0&&!emitted[cell]) {
            int rc=emit(cell,now);if(rc)return rc;
        }
        rate[cell]=0;
    }
    for(int cell=0;cell<cells;cell++) {
        if(capacity[cell]>0) {
            capacity[cell]-=rate[cell]*(duration-last[cell]);
            if(capacity[cell]<0)capacity[cell]=0;
        }
        absorbed[cell]=initial_capacity[cell]-capacity[cell];
        if(source[cell]>0&&!emitted[cell]) {
            source_absorbed[cell]+=source[cell]*duration;
            local_absorbed+=source[cell]*duration;
        }
        for(int ri=head[cell];ri>=0;ri=rays[ri].next) {
            double amount=rays[ri].rate*(duration-rays[ri].arrival);
            source_absorbed[rays[ri].source]+=amount;
            angle_absorbed[rays[ri].direction]+=amount;
        }
    }
    for(int k=0;k<nd;k++)angle_absorbed[k]+=local_absorbed/nd;
    statistics[0]=events;statistics[1]=rays.size();statistics[2]=queries;
    statistics[3]=assignments;statistics[4]=stale;statistics[5]=sizeof(Ray);
    last_heap_peak=heap.peak;last_heap_updates=heap.updates;
    last_ray_bytes=rays.allocated_bytes();
    return 0;
}
