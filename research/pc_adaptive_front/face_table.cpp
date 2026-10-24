/* Lazy exact realization of the uniform eight-subcell gas model.
 * A parent evolves analytically while its eight children remain identical.
 * The first arriving materialized ray promotes it; cleared children never
 * become opaque. Fully cleared parents release their child slot after all
 * incident rays and source octants have been processed.
 */
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#ifndef PC_HEAP_ARITY
#define PC_HEAP_ARITY 4
#endif
#ifndef PC_COMPACT_RAYS
#define PC_COMPACT_RAYS 1
#endif
#ifndef PC_ALIGN_NODES
#define PC_ALIGN_NODES 0
#endif
static_assert(PC_HEAP_ARITY >= 2, "Heap arity must be at least two");
#ifndef PC_FACE_TABLE_MAX_BYTES
#define PC_FACE_TABLE_MAX_BYTES (64*1024*1024)
#endif
#ifndef PC_FACE_TABLE_MIN_SOURCES
#define PC_FACE_TABLE_MIN_SOURCES 16
#endif
namespace {
struct alignas(PC_ALIGN_NODES ? 64 : alignof(double)) Node {
    double capacity=0, low=0, absorbed=0, compensation=0;
    double rate=0, last=0, initial=0;
    int head=-1;
    bool emitted=false, processed=false;
};
static_assert(sizeof(Node)==64, "Alignment must preserve node storage size");
struct Parent {int slot=-1; uint8_t opaque=8, pending=8;};
struct WideRay {
    long double travelled;
    double rate, arrival;
    int voxel[3], source, direction, next;
};
struct CompactRay {
    long double travelled;
    double rate, arrival;
    int source, next;
    int16_t voxel[3];
    uint16_t direction;
};
static_assert(sizeof(CompactRay)==48, "Expected 48-byte compact rays");
static_assert(sizeof(WideRay)==64, "Expected 64-byte wide rays");
template<class Ray> class RayStore {
    static constexpr size_t block_size=4096;
    std::vector<std::unique_ptr<Ray[]>> blocks;
    size_t count=0;
public:
    size_t size() const {return count;}
    size_t bytes() const {return blocks.size()*block_size*sizeof(Ray);}
    Ray &operator[](size_t at) {return blocks[at/block_size][at%block_size];}
    void add(const Ray &ray) {
        if(count%block_size==0)blocks.emplace_back(new Ray[block_size]);
        (*this)[count++]=ray;
    }
};
class ChildPool {
    static constexpr size_t block_slots=64;
    std::vector<std::unique_ptr<Node[]>> blocks;
    std::vector<int> unused;
public:
    std::vector<int> owner;
    size_t live=0, peak=0;
    int allocate(int parent) {
        int slot;
        if(unused.empty()) {
            slot=(int)owner.size();
            if((size_t)slot%block_slots==0)blocks.emplace_back(new Node[block_slots*8]);
            owner.push_back(parent);
        } else {slot=unused.back();unused.pop_back();owner[slot]=parent;}
        for(int j=0;j<8;j++)at(slot,j)=Node{};
        live++;peak=std::max(peak,live);
        return slot;
    }
    Node &at(int slot,int child) {return blocks[slot/block_slots][(slot%block_slots)*8+child];}
    void release(int slot) {owner[slot]=-1;unused.push_back(slot);live--;}
    size_t bytes() const {
        return blocks.size()*block_slots*8*sizeof(Node)+owner.capacity()*sizeof(int)+unused.capacity()*sizeof(int);
    }
};
struct Event {double time; int node;};
class Heap {
    std::vector<Event> data;
    std::vector<int> position;
    static bool earlier(const Event&a,const Event&b) {
        return a.time<b.time||(a.time==b.time&&a.node<b.node);
    }
    void exchange(int a,int b) {
        std::swap(data[a],data[b]);position[data[a].node]=a;position[data[b].node]=b;
    }
    void up(int at) {
        while(at>0){int p=(at-1)/PC_HEAP_ARITY;if(!earlier(data[at],data[p]))break;exchange(at,p);at=p;}
    }
    void down(int at) {
        const int size=(int)data.size();
        while((int64_t)PC_HEAP_ARITY*at+1<size) {
            int first=PC_HEAP_ARITY*at+1,c=first;
            int end=(int)std::min((int64_t)size,(int64_t)first+PC_HEAP_ARITY);
            for(int other=first+1;other<end;other++)
                if(earlier(data[other],data[c]))c=other;
            if(!earlier(data[c],data[at]))break;
            exchange(at,c);at=c;
        }
    }
public:
    explicit Heap(int count):position(count,-1){}
    void reserve_nodes(size_t count) {if(position.size()<count)position.resize(count,-1);}
    bool empty() const{return data.empty();}
    Event top() const{return data.front();}
    void erase(int node) {
        int at=position[node];if(at<0)return;
        position[node]=-1;
        if(at==(int)data.size()-1){data.pop_back();return;}
        data[at]=data.back();data.pop_back();position[data[at].node]=at;
        if(at>0&&earlier(data[at],data[(at-1)/PC_HEAP_ARITY]))up(at);else down(at);
    }
    void update(int node,double time) {
        int at=position[node];
        if(at<0){position[node]=(int)data.size();data.push_back({time,node});up((int)data.size()-1);}
        else {double old=data[at].time;data[at].time=time;if(time<old)up(at);else down(at);}
    }
    void pop(){erase(data.front().node);}
    size_t bytes() const{return data.capacity()*sizeof(Event)+position.capacity()*sizeof(int);}
};
struct Packed {
    int parent;
    std::array<double,8> remaining,absorbed;
};
struct InitialPacked {int parent;std::array<double,8> remaining;};
thread_local int64_t last_face_table_bytes=0;
thread_local std::vector<Packed> last_state;
thread_local std::vector<InitialPacked> last_initial;
thread_local std::vector<double> last_absorbed,last_remaining;

template<class Ray> int evolve(int n,const double*source,const double*initial_remaining,
           int n_sparse,const int32_t*sparse_parent,const double*sparse_remaining,
           int nl,const int32_t*source_level,const int32_t*level_offsets,
           const int32_t*level_counts,int nd,const double*directions,
           double duration,double max_distance,int periodic,
           double*absorbed,double*source_absorbed,double*source_escaped,double*source_retained,
           double*angle_absorbed,double*angle_escaped,double*angle_retained,double*statistics) {
    last_state.clear();last_initial.clear();last_absorbed.clear();last_remaining.clear();
    last_face_table_bytes=0;
    if(n<1||nl<1||nd<1||n_sparse<0||!std::isfinite(duration)||duration<=0||
       !std::isfinite(max_distance)||max_distance<=0)return 1;
    if((int64_t)n*n>INT_MAX/n||2.L*(max_distance+n+2)>INT_MAX)return 2;
    const int cells=n*n*n,fn=2*n;
    if(n_sparse>cells)return 3;
    for(int l=0;l<nl;l++)
        if(level_counts[l]<8||level_counts[l]%8||level_offsets[l]<0||
           (int64_t)level_offsets[l]+level_counts[l]>nd)return 4;
    for(int64_t d=0;d<3*(int64_t)nd;d++)if(!std::isfinite(directions[d]))return 5;
    std::vector<Node> coarse(cells);
    std::vector<Parent> parents(cells);
    ChildPool children;RayStore<Ray> rays;Heap heap(cells);
    std::vector<double> local_absorbed((size_t)nl*8,0.);
    std::vector<long double> face_table;
    size_t face_width=0;
    std::vector<uint8_t> initially_packed(cells,0);
    int64_t events=0,queries=0,assignments=0,promotions=0,parent_skips=0;
    int64_t uniform_events=0,source_launches=0,fine_events=0,coarsenings=0;
    size_t peak_cell_bytes=0,peak_heap_bytes=0;
    auto observe_memory=[&]() {
        peak_cell_bytes=std::max(peak_cell_bytes,coarse.capacity()*sizeof(Node)+
            parents.capacity()*sizeof(Parent)+children.bytes()+local_absorbed.capacity()*sizeof(double)+
            initially_packed.capacity()*sizeof(uint8_t)+last_initial.capacity()*sizeof(InitialPacked)+
            last_state.capacity()*sizeof(Packed)+(last_absorbed.capacity()+last_remaining.capacity())*sizeof(double)+
            face_table.capacity()*sizeof(long double));
        peak_heap_bytes=std::max(peak_heap_bytes,heap.bytes());
    };
    auto node=[&](int id)->Node& {
        return id<cells?coarse[id]:children.at((id-cells)/8,(id-cells)%8);
    };
    auto cell_of=[&](int id)->int {return id<cells?id:children.owner[(id-cells)/8];};
    auto fine_id=[&](int slot,int child)->int {return cells+8*slot+child;};
    auto schedule=[&](int id,double now) {
        Node &a=node(id);
        if(a.capacity>0&&a.rate>0) {
            double time=now+(a.capacity+a.low)/a.rate;
            if(time<duration)heap.update(id,time);else heap.erase(id);
        } else heap.erase(id);
    };
    auto consume=[&](int id,double now) {
        Node &a=node(id);double elapsed=now-a.last;a.last=now;
        if(elapsed<=0||a.rate<=0||a.capacity<=0)return;
        double amount=a.rate*elapsed;
        if(amount>=a.capacity+a.low) {
            a.absorbed=a.initial;a.compensation=0;
            if(id>=cells)parents[cell_of(id)].opaque--;
            a.capacity=a.low=0;return;
        }
        double y=amount-a.compensation,total=a.absorbed+y;
        a.compensation=(total-a.absorbed)-y;a.absorbed=total;
        double hi=a.capacity,delta=-amount,sum=hi+delta,part=sum-hi;
        double error=(hi-(sum-part))+(delta-part),low=a.low+error,next=sum+low;
        a.low=low-(next-sum);a.capacity=next;
    };
    auto allocate=[&](int parent)->int {
        if(children.owner.size()>=(size_t)(INT_MAX-cells)/8)return -1;
        int slot=children.allocate(parent);
        heap.reserve_nodes((size_t)cells+8*children.owner.size());
        parents[parent].slot=slot;
        observe_memory();return slot;
    };
    int active_sources=0;
    for(int p=0;p<cells;p++) {
        if(!std::isfinite(source[p])||source[p]<0||!std::isfinite(initial_remaining[p])||
           initial_remaining[p]<0||source_level[p]<0||source_level[p]>=nl||
           (initial_remaining[p]>0&&initial_remaining[p]/8==0))return 6;
        coarse[p].capacity=coarse[p].initial=initial_remaining[p];coarse[p].rate=source[p];
        if(source[p]>0)active_sources++;
        if(initial_remaining[p]<=0){parents[p].slot=-2;parents[p].opaque=parents[p].pending=0;}
    }
    // Every source origin and voxel face is an integer in fine-grid units.
    // Cache exactly the same division result, not an accumulated reciprocal.
    long double extent=std::ceil(2.L*max_distance)+2;
    if(active_sources>=PC_FACE_TABLE_MIN_SOURCES&&extent<INT_MAX&&
       3.L*nd*(extent+1)*sizeof(long double)<=PC_FACE_TABLE_MAX_BYTES) {
        face_width=(size_t)extent+1;
        face_table.resize((size_t)3*nd*face_width);
        for(int64_t axis=0;axis<3*(int64_t)nd;axis++) {
            long double speed=std::abs((long double)directions[axis]);
            for(size_t m=0;m<face_width;m++)
                face_table[(size_t)axis*face_width+m]=speed>0?(long double)m/speed:
                    std::numeric_limits<long double>::infinity();
        }
        last_face_table_bytes=(int64_t)(face_table.capacity()*sizeof(long double));
    }
    for(int k=0;k<n_sparse;k++) {
        int p=sparse_parent[k];
        if(p<0||p>=cells||initially_packed[p])return 7;
        initially_packed[p]=1;
        InitialPacked saved;saved.parent=p;long double sum=0;
        for(int j=0;j<8;j++) {
            double c=sparse_remaining[8*(size_t)k+j];
            if(!std::isfinite(c)||c<0)return 8;
            saved.remaining[j]=c;sum+=c;
        }
        if(std::abs(sum-(long double)initial_remaining[p])>
           2e-12L*std::max(sum,(long double)std::numeric_limits<double>::min()))return 9;
        last_initial.push_back(saved);
        if(sum==0)continue;
        bool equal=true;for(int j=1;j<8;j++)equal &= saved.remaining[j]==saved.remaining[0];
        if(equal)continue;
        int slot=allocate(p);if(slot<0)return 10;
        parents[p].opaque=parents[p].pending=0;
        for(int j=0;j<8;j++) {
            Node &a=children.at(slot,j);a.capacity=a.initial=saved.remaining[j];a.rate=source[p]/8;
            a.processed=a.capacity<=0;
            if(a.capacity>0){parents[p].opaque++;parents[p].pending++;}
        }
    }
    for(int p=0;p<cells;p++) {
        if(parents[p].slot<0)schedule(p,0);
        else for(int j=0;j<8;j++)schedule(fine_id(parents[p].slot,j),0);
    }
    observe_memory();
    auto promote=[&](int p,double now)->int {
        consume(p,now);
        if(coarse[p].capacity<=0) {
            parents[p].slot=-2;parents[p].opaque=parents[p].pending=0;
            // Keep the source's zero-duration exhaustion event: its remaining
            // emission must still be launched at this same physical time.
            if(now<duration)heap.update(p,now);
            return -2;
        }
        heap.erase(p);
        int slot=allocate(p);if(slot<0)return -3;
        const Node &base=coarse[p];
        parents[p].opaque=parents[p].pending=8;
        for(int j=0;j<8;j++) {
            Node &a=children.at(slot,j);
            a.capacity=base.capacity/8;a.low=base.low/8;a.initial=base.initial/8;
            a.absorbed=base.absorbed/8;a.compensation=base.compensation/8;
            a.rate=source[p]/8;a.last=now;
            schedule(fine_id(slot,j),now);
        }
        promotions++;observe_memory();return slot;
    };
    auto advance=[&](int ri,double now)->int {
        Ray &ray=rays[ri];const double*v=directions+3*ray.direction;
        const int origin[3]={2*(ray.source/(n*n))+1,
            2*((ray.source/n)%n)+1,2*(ray.source%n)+1};
        const long double horizon=2.L*max_distance;
        auto face_time=[&](int face,int axis)->long double {
            int64_t offset=(int64_t)face-origin[axis];
            uint64_t magnitude=offset<0?(uint64_t)(-offset):(uint64_t)offset;
            if(face_width&&magnitude<face_width) {
                long double saved=face_table[((size_t)3*ray.direction+axis)*face_width+(size_t)magnitude];
                return (offset<0)!=(v[axis]<0)?-saved:saved;
            }
            return (long double)offset/(long double)v[axis];
        };
        int answer=-1;
        for(int count=0;count<1000000;count++) {
            queries++;bool outside=false;int ijk[3];
            for(int a=0;a<3;a++) {
                outside|=ray.voxel[a]<0||ray.voxel[a]>=fn;
                ijk[a]=ray.voxel[a]%fn;if(ijk[a]<0)ijk[a]+=fn;
            }
            if(outside&&!periodic){answer=ray.travelled<=horizon?-1:-2;break;}
            if(ray.travelled>=horizon){answer=-2;break;}
            int p=((ijk[0]/2)*n+ijk[1]/2)*n+ijk[2]/2;
            int slot=parents[p].slot;
            if(slot==-1) {
                slot=promote(p,now);if(slot==-3)return 10;
            }
            if(slot>=0) {
                int child=(ijk[0]%2)*4+(ijk[1]%2)*2+ijk[2]%2;
                int id=fine_id(slot,child);
                if(node(id).capacity>0){answer=id;break;}
            }
            if(slot==-2||parents[p].opaque==0) {
                parent_skips++;
                long double crossing[3],next=std::numeric_limits<long double>::infinity();
                int base[3];
                for(int a=0;a<3;a++) {
                    base[a]=2*(ray.voxel[a]/2-(ray.voxel[a]<0&&ray.voxel[a]%2!=0));
                    crossing[a]=std::numeric_limits<long double>::infinity();
                    if(v[a]!=0){int face=base[a]+(v[a]>0?2:0);
                        crossing[a]=face_time(face,a);next=std::min(next,crossing[a]);}
                }
                if(!std::isfinite(next)||next<ray.travelled)return 11;
                if(next>horizon){answer=-2;break;}
                for(int a=0;a<3;a++) {
                    if(crossing[a]==next)ray.voxel[a]=base[a]+(v[a]>0?2:-1);
                    else if(v[a]!=0) {
                        long double mid=face_time(base[a]+1,a);
                        bool crossed=mid<=next;ray.voxel[a]=base[a]+(v[a]>0?crossed:!crossed);
                    }
                }
                ray.travelled=next;
            } else {
                // The current child is transparent but its parent is not.
                // Cross the at most three internal midplanes locally. Keep
                // absolute DDA face times, updating only axes that crossed;
                // no repeated additions of a reciprocal step are introduced.
                int base[3];long double crossing[3];
                for(int a=0;a<3;a++) {
                    base[a]=2*(ray.voxel[a]/2-(ray.voxel[a]<0&&ray.voxel[a]%2!=0));
                    crossing[a]=std::numeric_limits<long double>::infinity();
                    if(v[a]!=0) {
                        int face=ray.voxel[a]+(v[a]>0?1:0);
                        crossing[a]=face_time(face,a);
                    }
                }
                bool exited=false;
                for(int inner=0;inner<4;inner++) {
                    long double next=std::numeric_limits<long double>::infinity();
                    for(int a=0;a<3;a++)next=std::min(next,crossing[a]);
                    if(!std::isfinite(next)||next<ray.travelled)return 12;
                    if(next>horizon){answer=-2;break;}
                    bool crossed[3];
                    for(int a=0;a<3;a++) {
                        crossed[a]=crossing[a]==next;
                        if(crossed[a])ray.voxel[a]+=v[a]>0?1:-1;
                    }
                    ray.travelled=next;
                    for(int a=0;a<3;a++)
                        exited|=ray.voxel[a]<base[a]||ray.voxel[a]>=base[a]+2;
                    if(exited)break; // Outer loop handles periodic/open faces.
                    queries++;
                    if(ray.travelled>=horizon){answer=-2;break;}
                    int octant=(ray.voxel[0]-base[0])*4+
                        (ray.voxel[1]-base[1])*2+ray.voxel[2]-base[2];
                    int child_id=fine_id(slot,octant);
                    if(node(child_id).capacity>0){answer=child_id;break;}
                    for(int a=0;a<3;a++)if(crossed[a]) {
                        int face=ray.voxel[a]+(v[a]>0?1:0);
                        crossing[a]=face_time(face,a);
                    }
                }
                if(answer>=0||answer==-2)break;
                if(!exited)return 18; // A straight ray crosses <=3 midplanes.
            }
            if(count==999999)return 13;
        }
        if(answer<0) {
            double future=ray.rate*(duration-now);
            if(answer==-1){source_escaped[ray.source]+=future;angle_escaped[ray.direction]+=future;}
            else {source_retained[ray.source]+=future;angle_retained[ray.direction]+=future;}
            ray.next=-1;return 0;
        }
        consume(answer,now);Node &a=node(answer);a.rate+=ray.rate;
        ray.arrival=now;ray.next=a.head;a.head=ri;assignments++;
        if(a.capacity<=0)heap.update(answer,now);else schedule(answer,now);
        return 0;
    };
    auto emit=[&](int p,int octant,int id,double now)->int {
        Node &a=node(id);a.emitted=true;source_launches++;
        int lev=source_level[p],total=level_counts[lev],count=octant<0?total:total/8;
        int offset=level_offsets[lev]+(octant<0?0:octant*count);
        double local=source[p]*now/(octant<0?1:8);
        source_absorbed[p]+=local;
        if(octant<0)for(int j=0;j<8;j++)local_absorbed[(size_t)lev*8+j]+=local/8;
        else local_absorbed[(size_t)lev*8+octant]+=local;
        int xyz[3]={2*(p/(n*n)),2*((p/n)%n),2*(p%n)};
        for(int k=0;k<count;k++) {
            if(rays.size()>=(size_t)INT_MAX)return 14;
            int oc=octant<0?k/(total/8):octant;
            Ray ray{};
            ray.travelled=0;ray.rate=source[p]/total;ray.arrival=now;
            ray.voxel[0]=xyz[0]+oc/4;ray.voxel[1]=xyz[1]+(oc/2)%2;ray.voxel[2]=xyz[2]+oc%2;
            ray.source=p;ray.direction=offset+k;ray.next=-1;rays.add(ray);
            int rc=advance((int)rays.size()-1,now);if(rc)return rc;
        }
        return 0;
    };
    for(int p=0;p<cells;p++)if(source[p]>0) {
        if(parents[p].slot==-2){int rc=emit(p,-1,p,0);if(rc)return rc;}
        else if(parents[p].slot>=0)for(int j=0;j<8;j++) {
            int id=fine_id(parents[p].slot,j);
            if(node(id).capacity<=0&&!node(id).emitted){int rc=emit(p,j,id,0);if(rc)return rc;}
        }
    }
    while(!heap.empty()) {
        Event event=heap.top();heap.pop();if(event.time>=duration)break;
        int id=event.node,p=cell_of(id);double now=event.time;
        if(p<0||p>=cells)return 15;
        Node &a=node(id);events++;
        if(id<cells) {
            if(parents[p].slot>=0)return 16;
            a.absorbed=a.initial;a.compensation=0;a.capacity=a.low=0;a.last=now;
            parents[p].slot=-2;parents[p].opaque=parents[p].pending=0;uniform_events++;
        } else {
            if(a.processed)return 17;
            if(a.capacity>0)parents[p].opaque--;
            a.absorbed=a.initial;a.compensation=0;a.capacity=a.low=0;a.last=now;fine_events++;
        }
        int ri=a.head;a.head=-1;
        while(ri>=0) {
            int next=rays[ri].next;double amount=rays[ri].rate*(now-rays[ri].arrival);
            source_absorbed[rays[ri].source]+=amount;angle_absorbed[rays[ri].direction]+=amount;
            int rc=advance(ri,now);if(rc)return rc;ri=next;
        }
        if(source[p]>0&&!a.emitted){int rc=emit(p,id<cells?-1:(id-cells)%8,id,now);if(rc)return rc;}
        a.rate=0;a.processed=true;
        if(id>=cells) {
            parents[p].pending--;
            if(parents[p].pending==0) {
                int slot=parents[p].slot;
                for(int j=0;j<8;j++)heap.erase(fine_id(slot,j));
                coarse[p].absorbed=coarse[p].initial;coarse[p].compensation=0;
                coarse[p].capacity=coarse[p].low=0;coarse[p].rate=0;
                coarse[p].emitted=true;coarse[p].processed=true;
                parents[p].slot=-2;children.release(slot);coarsenings++;
            }
        }
        observe_memory();
    }
    last_absorbed.resize(cells);last_remaining.resize(cells);
    auto finish=[&](int id,int p,int octant) {
        consume(id,duration);Node&a=node(id);
        if(source[p]>0&&!a.emitted) {
            double local=source[p]*duration/(octant<0?1:8);source_absorbed[p]+=local;
            int lev=source_level[p];
            if(octant<0)for(int j=0;j<8;j++)local_absorbed[(size_t)lev*8+j]+=local/8;
            else local_absorbed[(size_t)lev*8+octant]+=local;
        }
        for(int ri=a.head;ri>=0;ri=rays[ri].next) {
            double amount=rays[ri].rate*(duration-rays[ri].arrival);
            source_absorbed[rays[ri].source]+=amount;angle_absorbed[rays[ri].direction]+=amount;
        }
    };
    for(int p=0;p<cells;p++) {
        int slot=parents[p].slot;
        if(slot<0) {
            finish(p,p,-1);absorbed[p]=coarse[p].absorbed;
            last_remaining[p]=coarse[p].capacity+coarse[p].low;
        } else {
            Packed state;state.parent=p;long double sum_a=0,sum_r=0;
            for(int j=0;j<8;j++) {
                int id=fine_id(slot,j);finish(id,p,j);Node&a=node(id);
                state.remaining[j]=a.capacity+a.low;state.absorbed[j]=a.absorbed;
                sum_a+=a.absorbed;sum_r+=state.remaining[j];
            }
            absorbed[p]=(double)sum_a;last_remaining[p]=(double)sum_r;
            if(sum_r>0)last_state.push_back(state);
        }
        last_absorbed[p]=absorbed[p];
    }
    for(int lev=0;lev<nl;lev++)for(int j=0;j<8;j++) {
        int count=level_counts[lev]/8,offset=level_offsets[lev]+j*count;
        for(int k=0;k<count;k++)angle_absorbed[offset+k]+=local_absorbed[(size_t)lev*8+j]/count;
    }
    observe_memory();
    statistics[0]=events;statistics[1]=rays.size();statistics[2]=queries;statistics[3]=assignments;
    statistics[4]=promotions;statistics[5]=children.peak;statistics[6]=children.owner.size();
    statistics[7]=rays.bytes();statistics[8]=peak_cell_bytes;statistics[9]=peak_heap_bytes;
    statistics[10]=parent_skips;statistics[11]=uniform_events;statistics[12]=last_state.size();
    statistics[13]=source_launches;statistics[14]=fine_events;statistics[15]=coarsenings;
    return 0;
}
} // namespace

extern "C" int adaptive_evolve(int n,const double*s,const double*c,int ns,const int32_t*sp,const double*sc,
    int nl,const int32_t*sl,const int32_t*lo,const int32_t*lc,int nd,const double*dir,
    double duration,double distance,int periodic,double*a,double*sa,double*se,double*sr,
    double*aa,double*ae,double*ar,double*stats) {
    try {
#if PC_COMPACT_RAYS
        bool compact=n>0&&nd>0&&nd<=65536&&std::isfinite(distance)&&distance>0&&
            2.L*n+2.L*distance+2<=INT16_MAX;
        if(compact)for(int64_t d=0;d<3*(int64_t)nd;d++)
            if(!std::isfinite(dir[d])||std::abs(dir[d])>1.)compact=false;
        if(compact)return evolve<CompactRay>(n,s,c,ns,sp,sc,nl,sl,lo,lc,nd,dir,
            duration,distance,periodic,a,sa,se,sr,aa,ae,ar,stats);
#endif
        return evolve<WideRay>(n,s,c,ns,sp,sc,nl,sl,lo,lc,nd,dir,
            duration,distance,periodic,a,sa,se,sr,aa,ae,ar,stats);
    }
    catch(const std::bad_alloc&){return 90;}
    catch(...){return 91;}
}
extern "C" int64_t adaptive_state_count(){return (int64_t)last_state.size();}
extern "C" int adaptive_copy_state(int32_t*parents,double*remaining) {
    for(size_t k=0;k<last_state.size();k++) {
        parents[k]=last_state[k].parent;
        for(int j=0;j<8;j++)remaining[8*k+j]=last_state[k].remaining[j];
    }
    return 0;
}
extern "C" int adaptive_copy_remaining(double*out) {
    std::copy(last_remaining.begin(),last_remaining.end(),out);return 0;
}
extern "C" int adaptive_copy_state_absorbed(double*out) {
    for(size_t k=0;k<last_state.size();k++)for(int j=0;j<8;j++)out[8*k+j]=last_state[k].absorbed[j];
    return 0;
}
extern "C" int adaptive_copy_fine_absorbed(double*out) {
    for(size_t p=0;p<last_absorbed.size();p++)for(int j=0;j<8;j++)out[8*p+j]=last_absorbed[p]/8;
    for(const auto &initial:last_initial)if(last_remaining[initial.parent]<=0)
        for(int j=0;j<8;j++)out[8*(size_t)initial.parent+j]=initial.remaining[j];
    for(const auto &state:last_state)
        for(int j=0;j<8;j++)out[8*(size_t)state.parent+j]=state.absorbed[j];
    return 0;
}

extern "C" int64_t adaptive_face_table_bytes(){return last_face_table_bytes;}
