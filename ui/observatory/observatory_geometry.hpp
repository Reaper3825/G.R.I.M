#pragma once
#include "observatory_report.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace GRIM::Observatory {
enum class VerticalAxis { Rank, Entropy };

// One scale for the whole position: changing top-k membership must not stretch
// individual layers. Candidates are probability-sorted by the report loader.
inline size_t rankCapacity(const PositionReadout& position) {
    size_t count=1;
    for(const auto& layer:position.layers)count=std::max(count,layer.candidates.size());
    return count;
}
inline double layerDepth(unsigned layer,unsigned count) {
    return count>1?2.0*layer/(count-1)-1.0:0.0;
}
inline std::array<double,3> candidateLocation(const Report& report,const LayerReadout& layer,
                                             size_t index,size_t ranks,VerticalAxis axis) {
    const double y=axis==VerticalAxis::Rank
        ? (ranks>1?1.0-2.0*index/(ranks-1):0.0)
        : 2.0*layer.entropyNats/std::max(1e-6,std::log(double(report.vocabSize)))-1.0;
    return {2.0*layer.candidates.at(index).probability-1.0,y,layerDepth(layer.layer,report.layerCount)};
}
using PathPoint=std::array<double,3>;
struct TokenTrajectory { int tokenId; std::vector<PathPoint> points; };
inline std::vector<TokenTrajectory> tokenTrajectories(const Report& report,const PositionReadout& position,VerticalAxis axis) {
    std::vector<TokenTrajectory> paths;
    std::map<int,std::pair<unsigned,size_t>> previous;
    const auto ranks=rankCapacity(position);
    for(const auto& layer:position.layers)for(size_t i=0;i<layer.candidates.size();++i) {
        const int id=layer.candidates[i].tokenId;
        auto found=previous.find(id);
        if(found==previous.end() || found->second.first+1!=layer.layer) {
            paths.push_back({id,{}});
            previous[id]={layer.layer,paths.size()-1};
        } else found->second.first=layer.layer;
        paths[previous[id].second].points.push_back(candidateLocation(report,layer,i,ranks,axis));
    }
    return paths;
}
// Shape-preserving cubic Hermite interpolation. Minmod tangents are shared at
// each knot, vanish at extrema, and cannot overshoot a segment's endpoint range.
// A path contains consecutive layers only, so its parameter spacing is uniform.
inline PathPoint sampleTrajectory(const TokenTrajectory& path,size_t segment,double t) {
    const auto& a=path.points.at(segment);
    const auto& b=path.points.at(segment+1);
    t=std::clamp(t,0.,1.);
    if(t==0)return a;
    if(t==1)return b;
    auto tangent=[&](size_t knot,size_t dimension) {
        if(knot==0)return path.points[1][dimension]-path.points[0][dimension];
        if(knot+1==path.points.size())return path.points[knot][dimension]-path.points[knot-1][dimension];
        const double left=path.points[knot][dimension]-path.points[knot-1][dimension];
        const double right=path.points[knot+1][dimension]-path.points[knot][dimension];
        return left*right>0?std::copysign(std::min(std::abs(left),std::abs(right)),left):0.;
    };
    const double t2=t*t,t3=t2*t;
    PathPoint result;
    for(size_t dim=0;dim<2;++dim) {
        result[dim]=(2*t3-3*t2+1)*a[dim]+(t3-2*t2+t)*tangent(segment,dim)
            +(-2*t3+3*t2)*b[dim]+(t3-t2)*tangent(segment+1,dim);
        result[dim]=std::clamp(result[dim],std::min(a[dim],b[dim]),std::max(a[dim],b[dim]));
    }
    result[2]=a[2]+t*(b[2]-a[2]); // Depth always advances linearly between real layers.
    return result;
}
struct CandidateChange {
    bool previousLayerAvailable=false;
    std::optional<double> probabilityDelta;
    std::optional<int> rankGain;
};
inline CandidateChange candidateChange(const PositionReadout& position,const LayerReadout& layer,size_t index) {
    CandidateChange change;
    if(layer.layer==0)return change;
    for(const auto& previous:position.layers)if(previous.layer==layer.layer-1) {
        change.previousLayerAvailable=!previous.candidates.empty();
        for(size_t i=0;i<previous.candidates.size();++i) {
            if(previous.candidates[i].tokenId!=layer.candidates.at(index).tokenId)continue;
            change.probabilityDelta=layer.candidates[index].probability-previous.candidates[i].probability;
            change.rankGain=int(i)-int(index);
            break;
        }
        break;
    }
    return change;
}
}
