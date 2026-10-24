/* Sparse resolved opaque fronts. Sources stay at original cell centres.
 * Only their eight touching fine cells receive analytic local emission.
 * Quadrature rays are allocated after those cells clear. Cleared cells need
 * one bit; illuminated opaque cells use recyclable hash-addressed nodes.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>
using Real=long double;
struct Ray {Real travelled=0,arrival=0,rate=0;int voxel[3],source,direction,next=-1;};
struct Node {Real initial=0,remaining=0,rate=0,last=0,used=0,local_rate=0;double old=0;int64_t key=-1;int head=-1,source=-1,octant=-1;};
struct Event {Real time;int node;};
class Heap {
 std::vector<Event> data;std::vector<int> position;
 static bool before(const Event&a,const Event&b){return a.time<b.time||(a.time==b.time&&a.node<b.node);}
 void exchange(int a,int b){std::swap(data[a],data[b]);position[data[a].node]=a;position[data[b].node]=b;}
 void up(int a){while(a){int p=(a-1)/4;if(!before(data[a],data[p]))break;exchange(a,p);a=p;}}
 void down(int a){while(4*a+1<(int)data.size()){int first=4*a+1,j=first,end=std::min(first+4,(int)data.size());for(int k=first+1;k<end;k++)if(before(data[k],data[j]))j=k;if(!before(data[j],data[a]))break;exchange(j,a);a=j;}}
 public:
 void ensure(size_t n){if(position.size()<n)position.resize(n,-1);}
 bool empty()const{return data.empty();}
 Event top()const{return data[0];}
 void erase(int id){int a=position[id];if(a<0)return;position[id]=-1;if(a==(int)data.size()-1){data.pop_back();return;}data[a]=data.back();data.pop_back();position[data[a].node]=a;if(a&&before(data[a],data[(a-1)/4]))up(a);else down(a);}
 void update(int id,Real t){int a=position[id];if(a<0){position[id]=data.size();data.push_back({t,id});up(data.size()-1);}else{Real old=data[a].time;data[a].time=t;if(t<old)up(a);else down(a);}}
 size_t bytes()const{return data.capacity()*sizeof(Event)+position.capacity()*sizeof(int);}
};
struct Segment {Real time,remaining,rate;};
struct LocalTemplate {
 int f,count;Real exit_time;
 std::vector<double> directions;
 std::vector<Real> clear_time;
 std::vector<std::vector<Segment>> history;
 Real absorbed(int cell,Real u)const{
   if(clear_time[cell]<=u)return Real(1)/(f*f*f);
   const auto&h=history[cell];auto it=std::upper_bound(h.begin(),h.end(),u,[](Real t,const Segment&a){return t<a.time;});
   if(it==h.begin())return 0;--it;
   return std::min(Real(1)/(f*f*f),(Real(1)/(f*f*f)-it->remaining)+it->rate*(u-it->time));
 }
 Real remaining(int cell,Real u)const{
   if(clear_time[cell]<=u)return 0;
   const auto&h=history[cell];auto it=std::upper_bound(h.begin(),h.end(),u,[](Real t,const Segment&a){return t<a.time;});
   if(it==h.begin())return Real(1)/(f*f*f);--it;
   return std::max(Real(0),it->remaining-it->rate*(u-it->time));
 }
};
thread_local std::vector<LocalTemplate> template_cache;
extern "C" void refined_clear_template_cache(){template_cache.clear();}
static LocalTemplate build_template(int f,int nk,const double*dirs){
 LocalTemplate out;out.f=f;out.count=nk;out.exit_time=std::numeric_limits<Real>::infinity();out.directions.assign(dirs,dirs+3*nk);
 const int nc=f*f*f;const Real cap=Real(1)/nc;
 out.clear_time.assign(nc,std::numeric_limits<Real>::infinity());out.history.resize(nc);
 std::vector<Real> rem(nc,cap),rate(nc,0),last(nc,0);std::vector<int> head(nc,-1);std::vector<Ray> rays(nk);Heap heap;heap.ensure(nc);
 auto record=[&](int cell,Real now){auto&h=out.history[cell];Segment s{now,rem[cell],rate[cell]};if(!h.empty()&&h.back().time==now)h.back()=s;else h.push_back(s);};
 auto attach=[&](int ri,Real now){Ray&r=rays[ri];const double*v=dirs+3*r.direction;
   for(;;){bool outside=false;for(int a=0;a<3;a++)outside|=r.voxel[a]<0||r.voxel[a]>=f;if(outside){out.exit_time=std::min(out.exit_time,now);return;}
     int cell=(r.voxel[0]*f+r.voxel[1])*f+r.voxel[2];
     if(out.clear_time[cell]>now){rem[cell]=std::max(Real(0),rem[cell]-rate[cell]*(now-last[cell]));last[cell]=now;rate[cell]+=r.rate;r.next=head[cell];head[cell]=ri;record(cell,now);heap.update(cell,now+rem[cell]/rate[cell]);return;}
     Real crossing[3],next=std::numeric_limits<Real>::infinity();for(int a=0;a<3;a++){crossing[a]=std::numeric_limits<Real>::infinity();if(v[a]){crossing[a]=((Real)r.voxel[a]+(v[a]>0?1:0)-f/2)/(Real)v[a];next=std::min(next,crossing[a]);}}
     for(int a=0;a<3;a++)if(crossing[a]==next)r.voxel[a]+=v[a]>0?1:-1;r.travelled=next;
   }
 };
 for(int d=0;d<nk;d++){Ray&r=rays[d];r.direction=d;r.rate=Real(1)/nk;for(int a=0;a<3;a++)r.voxel[a]=f/2-(dirs[3*d+a]<0?1:0);attach(d,0);}
 while(!heap.empty()&&heap.top().time<=out.exit_time){Event e=heap.top();heap.erase(e.node);int cell=e.node;Real now=e.time;rem[cell]=0;rate[cell]=0;last[cell]=now;out.clear_time[cell]=now;record(cell,now);int ri=head[cell];head[cell]=-1;while(ri>=0){int next=rays[ri].next;attach(ri,now);ri=next;}}
 return out;
}

thread_local std::vector<int64_t> state_ids;
thread_local std::vector<double> state_ionized;
extern "C" int64_t refined_state_count(){return state_ids.size();}
extern "C" void refined_copy_state(int64_t*ids,double*ionized){std::copy(state_ids.begin(),state_ids.end(),ids);std::copy(state_ionized.begin(),state_ionized.end(),ionized);}

extern "C" int refined_evolve(int n,int f,const double*photons,const double*capacity,
 const int32_t*levels,const int32_t*offsets,const int32_t*counts,const double*directions,
 int64_t np,const int64_t*initial_ids,const double*initial_ionized,uint64_t*clear,
 double horizon,int periodic,int use_templates,double*absorbed,double*source_absorbed,double*escaped,double*retained,
 double*fine_absorbed,double*stats){
 state_ids.clear();state_ionized.clear();
 if(n<1||(f!=2&&f!=4&&f!=8)||np<0||!std::isfinite(horizon)||horizon<=0)return 1;
 const int64_t nc64=(int64_t)n*n*n;if(nc64>std::numeric_limits<int>::max()||(int64_t)n*f>std::numeric_limits<int>::max())return 2;
 const int nc=(int)nc64,fn=n*f,sub=f*f*f,words=(sub+63)/64;
 const uint64_t full_last=sub<64?((uint64_t(1)<<sub)-1):~uint64_t(0);
 std::vector<Node> nodes;std::vector<int> free_ids;std::unordered_map<int64_t,int> lookup;
 std::vector<Ray> rays;Heap heap;
 std::vector<Real> used_coarse(nc,0),used_source(nc,0),escape_source(nc,0),retain_source(nc,0);
 int64_t events=0,queries=0,assignments=0,created=0,peak_live=0,parent_skips=0,launches=0;
 int64_t template_parents=0,materialisations=0,implicit_parents=0,builds=0,implicit_directions=0;double template_build_seconds=0;
 std::vector<uint8_t> isolated(nc,0);std::vector<int> parent_template(nc,-1);std::vector<Event> isolated_events;size_t isolated_cursor=0;
 std::vector<std::pair<int,Real>> pending_launches;size_t pending_cursor=0;
 auto geometry=[&](int64_t key,int &p,int &child){int z=key%fn,y=(key/fn)%fn,x=key/((int64_t)fn*fn);p=((x/f)*n+y/f)*n+z/f;child=((x%f)*f+y%f)*f+z%f;};
 auto full=[&](int p){if(capacity[p]==0)return true;for(int w=0;w<words;w++)if(clear[(size_t)p*words+w]!=(w==words-1?full_last:~uint64_t(0)))return false;return true;};
 auto cleared=[&](int p,int child){return capacity[p]==0||((clear[(size_t)p*words+child/64]>>(child%64))&1);};
 auto new_node=[&](int64_t key,double old)->int{
   int p,child;geometry(key,p,child);int id;
   if(free_ids.empty()){if(nodes.size()>=(size_t)std::numeric_limits<int>::max())return -1;id=nodes.size();nodes.push_back(Node{});heap.ensure(nodes.size());}
   else{id=free_ids.back();free_ids.pop_back();nodes[id]=Node{};}
   Node &a=nodes[id];a.key=key;a.old=old;a.initial=a.remaining=(Real)(capacity[p]/sub)-(Real)old;
   lookup[key]=id;created++;peak_live=std::max(peak_live,(int64_t)lookup.size());return id;
 };
 for(int64_t j=0;j<np;j++){int id=new_node(initial_ids[j],initial_ionized[j]);if(id<0)return 3;}
 if(use_templates&&horizon>=f*std::sqrt(3.L)/2){
   std::vector<uint8_t> nonneutral(nc,0);for(int64_t j=0;j<np;j++){int p,child;geometry(initial_ids[j],p,child);nonneutral[p]=1;}
   int nl=1;for(int p=0;p<nc;p++)nl=std::max(nl,levels[p]+1);std::vector<int> level_templates(nl,-1);
   for(int p=0;p<nc;p++){
     if(photons[p]<=0||capacity[p]<=0||nonneutral[p])continue;
     bool neutral=true;for(int w=0;w<words;w++)neutral&=clear[(size_t)p*words+w]==0;if(!neutral)continue;
     int nk=counts[levels[p]],offset=offsets[levels[p]],ti=level_templates[levels[p]];
     if(ti<0)for(size_t j=0;j<template_cache.size();j++)if(template_cache[j].f==f&&template_cache[j].count==nk&&std::equal(template_cache[j].directions.begin(),template_cache[j].directions.end(),directions+3*offset)){ti=j;break;}
     if(ti<0){auto before=std::chrono::steady_clock::now();template_cache.push_back(build_template(f,nk,directions+3*offset));ti=template_cache.size()-1;builds++;template_build_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();}
     level_templates[levels[p]]=ti;parent_template[p]=ti;isolated[p]=1;template_parents++;
     Real when=template_cache[ti].exit_time*(Real)capacity[p]/(Real)photons[p];if(when<1)isolated_events.push_back({when,p});
   }
   std::sort(isolated_events.begin(),isolated_events.end(),[](const Event&a,const Event&b){return a.time<b.time||(a.time==b.time&&a.node<b.node);});
 }
 auto install_template=[&](int p,Real now,bool materialise)->int{
   const LocalTemplate&t=template_cache[parent_template[p]];Real u=now*(Real)photons[p]/(Real)capacity[p];
   used_source[p]+=(Real)photons[p]*now;
   int px=p/(n*n),py=(p/n)%n,pz=p%n;
   for(int child=0;child<sub;child++){
     Real rem=t.remaining(child,u);Real amount=((Real)capacity[p])*t.absorbed(child,u);
     int x=px*f+child/(f*f),y=py*f+(child/f)%f,z=pz*f+child%f;int64_t key=((int64_t)x*fn+y)*fn+z;
     used_coarse[p]+=amount;if(fine_absorbed)fine_absorbed[key]+=(double)amount;
     if(rem==0){clear[(size_t)p*words+child/64]|=uint64_t(1)<<(child%64);continue;}
     if(amount>0){if(materialise){auto found=lookup.find(key);int id=found==lookup.end()?new_node(key,(double)amount):found->second;if(id<0)return 3;nodes[id].last=now;}else{state_ids.push_back(key);state_ionized.push_back((double)amount);}}
   }
   return 0;
 };
 auto promote=[&](int p,Real now)->int{
   if(!isolated[p])return 0;isolated[p]=0;materialisations++;int rc=install_template(p,now,true);if(rc)return rc;pending_launches.push_back({p,now});return 0;
 };
 auto consume=[&](int id,Real now){Node&a=nodes[id];if(now<a.last)return false;Real amount=std::min(a.remaining,a.rate*(now-a.last));a.remaining-=amount;a.used+=amount;a.last=now;return true;};
 auto schedule=[&](int id,Real now){Node&a=nodes[id];if(a.rate>0){Real time=now+a.remaining/a.rate;if(time<1)heap.update(id,time);else heap.erase(id);}else heap.erase(id);};
 auto origin=[&](int source,int axis)->Real{int coord=axis==0?source/(n*n):(axis==1?(source/n)%n:source%n);return (Real)(coord*f)+f/2;};
 auto floor_div=[](int a,int b){int q=a/b;if(a<0&&a%b)q--;return q;};
 auto attach=[&](int ri,Real now)->int{
   for(int64_t steps=0;steps<100000000;steps++){
     Ray&r=rays[ri];const double*v=directions+3*r.direction;queries++;
     bool outside=false;int ijk[3];for(int a=0;a<3;a++){outside|=r.voxel[a]<0||r.voxel[a]>=fn;ijk[a]=r.voxel[a]%fn;if(ijk[a]<0)ijk[a]+=fn;}
     if(outside&&!periodic){escape_source[r.source]+=r.rate*(1-now);return 0;}
     if(r.travelled>=(Real)horizon){retain_source[r.source]+=r.rate*(1-now);return 0;}
     int p=((ijk[0]/f)*n+ijk[1]/f)*n+ijk[2]/f,child=((ijk[0]%f)*f+ijk[1]%f)*f+ijk[2]%f;
     if(isolated[p]){int rc=promote(p,now);if(rc)return rc;}
     bool skip=full(p);
     if(!cleared(p,child)){
       int64_t key=((int64_t)ijk[0]*fn+ijk[1])*fn+ijk[2];auto found=lookup.find(key);int id=found==lookup.end()?new_node(key,0):found->second;if(id<0)return 3;
       if(!consume(id,now))return 4;Node&a=nodes[id];a.rate+=r.rate;r.arrival=now;r.next=a.head;a.head=ri;assignments++;schedule(id,now);return 0;
     }
     Real crossing[3],next=std::numeric_limits<Real>::infinity();int boundary[3];
     for(int a=0;a<3;a++){
       crossing[a]=std::numeric_limits<Real>::infinity();boundary[a]=r.voxel[a];
       if(v[a]!=0){boundary[a]=skip?(floor_div(r.voxel[a],f)+(v[a]>0?1:0))*f:r.voxel[a]+(v[a]>0?1:0);crossing[a]=((Real)boundary[a]-origin(r.source,a))/(Real)v[a];next=std::min(next,crossing[a]);}
     }
     if(!std::isfinite(next)||next<r.travelled)return 5;
     if(next>(Real)horizon){retain_source[r.source]+=r.rate*(1-now);return 0;}
     if(skip){parent_skips++;for(int a=0;a<3;a++){
       Real x=origin(r.source,a)+next*(Real)v[a];int b=(int)std::floor(x);if(v[a]<0&&x==std::floor(x))b--;
       r.voxel[a]=crossing[a]==next?boundary[a]-(v[a]<0?1:0):b;
     }}else for(int a=0;a<3;a++)if(crossing[a]==next)r.voxel[a]+=v[a]>0?1:-1;
     r.travelled=next;
   }return 6;
 };
 auto launch=[&](int source,int octant,Real now)->int{
   launches++;const int count=counts[levels[source]],start=offsets[levels[source]]+octant*(count/8);
   for(int d=start;d<start+count/8;d++){
     if(rays.size()>=(size_t)std::numeric_limits<int>::max())return 7;
     Ray r;r.source=source;r.direction=d;r.rate=(Real)photons[source]/count;
     for(int a=0;a<3;a++)r.voxel[a]=(int)origin(source,a)-(directions[3*d+a]<0?1:0);
     rays.push_back(r);int rc=attach((int)rays.size()-1,now);if(rc)return rc;
   }return 0;
 };
 auto drain_launches=[&]()->int{while(pending_cursor<pending_launches.size()){auto item=pending_launches[pending_cursor++];for(int oct=0;oct<8;oct++){int rc=launch(item.first,oct,item.second);if(rc)return rc;}}pending_launches.clear();pending_cursor=0;return 0;};
 for(int s=0;s<nc;s++)if(photons[s]>0&&parent_template[s]<0)for(int oct=0;oct<8;oct++){
   int xyz[3];for(int a=0;a<3;a++)xyz[a]=(int)origin(s,a)-(((oct>>(2-a))&1)?0:1);
   int64_t key=((int64_t)xyz[0]*fn+xyz[1])*fn+xyz[2];int p,child;geometry(key,p,child);
   if(cleared(p,child)){int rc=launch(s,oct,0);if(rc)return rc;continue;}
   auto found=lookup.find(key);int id=found==lookup.end()?new_node(key,0):found->second;if(id<0)return 3;
   Node&a=nodes[id];a.source=s;a.octant=oct;a.local_rate=((Real)photons[s]/counts[levels[s]])*(counts[levels[s]]/8);a.rate+=a.local_rate;schedule(id,0);
 }
 {int rc=drain_launches();if(rc)return rc;}
 for(;;){
   while(isolated_cursor<isolated_events.size()&&!isolated[isolated_events[isolated_cursor].node])isolated_cursor++;
   if(isolated_cursor<isolated_events.size()&&(heap.empty()||isolated_events[isolated_cursor].time<=heap.top().time)){
     Event e=isolated_events[isolated_cursor++];int rc=promote(e.node,e.time);if(rc)return rc;rc=drain_launches();if(rc)return rc;continue;
   }
   if(heap.empty())break;Event e=heap.top();heap.erase(e.node);if(e.time>=1)break;const Real now=e.time;
   Node a=nodes[e.node];int p,child;geometry(a.key,p,child);events++;
   a.used=a.initial;used_coarse[p]+=a.used;if(fine_absorbed)fine_absorbed[a.key]+=(double)a.used;
   clear[(size_t)p*words+child/64]|=uint64_t(1)<<(child%64);
   lookup.erase(a.key);nodes[e.node].key=-1;free_ids.push_back(e.node);
   if(a.source>=0){used_source[a.source]+=a.local_rate*now;int rc=launch(a.source,a.octant,now);if(rc)return rc;}
   for(int ri=a.head;ri>=0;){int next=rays[ri].next;used_source[rays[ri].source]+=rays[ri].rate*(now-rays[ri].arrival);int rc=attach(ri,now);if(rc)return rc;ri=next;}
   {int rc=drain_launches();if(rc)return rc;}
 }
 for(int p=0;p<nc;p++)if(isolated[p]){int rc=install_template(p,1,false);if(rc)return rc;implicit_parents++;implicit_directions+=counts[levels[p]];}
 for(size_t id=0;id<nodes.size();id++)if(nodes[id].key>=0){
   if(!consume(id,1))return 4;Node&a=nodes[id];int p,child;geometry(a.key,p,child);used_coarse[p]+=a.used;if(fine_absorbed)fine_absorbed[a.key]+=(double)a.used;
   if(a.source>=0)used_source[a.source]+=a.local_rate;
   for(int ri=a.head;ri>=0;ri=rays[ri].next)used_source[rays[ri].source]+=rays[ri].rate*(1-rays[ri].arrival);
   double cumulative=(double)((Real)a.old+a.used);
   if(a.remaining==0)clear[(size_t)p*words+child/64]|=uint64_t(1)<<(child%64);
   else if(cumulative>0){state_ids.push_back(a.key);state_ionized.push_back(cumulative);}
 }
 for(int p=0;p<nc;p++){absorbed[p]=(double)used_coarse[p];source_absorbed[p]=(double)used_source[p];escaped[p]=(double)escape_source[p];retained[p]=(double)retain_source[p];}
 stats[0]=events;stats[1]=queries;stats[2]=assignments;stats[3]=rays.size();stats[4]=rays.capacity()*sizeof(Ray);stats[5]=created;stats[6]=peak_live;stats[7]=nodes.capacity()*sizeof(Node)+free_ids.capacity()*sizeof(int);stats[8]=heap.bytes();stats[9]=lookup.bucket_count()*sizeof(void*)+peak_live*(sizeof(std::pair<const int64_t,int>)+2*sizeof(void*));stats[10]=parent_skips;stats[11]=launches;stats[12]=template_parents;stats[13]=materialisations;stats[14]=implicit_parents;stats[15]=builds;stats[16]=template_build_seconds;stats[17]=implicit_directions;
 return 0;
}
