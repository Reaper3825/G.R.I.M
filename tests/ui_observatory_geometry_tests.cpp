#include "ui/observatory/observatory_geometry.hpp"
#include <iostream>
#include <stdexcept>
using namespace GRIM::Observatory;
namespace {
void check(bool value) {if(!value)throw std::runtime_error("Observatory geometry assertion failed");}
bool near(double a,double b) {return std::abs(a-b)<1e-9;}
}
int main() {
    Report report;report.layerCount=4;report.vocabSize=100;
    PositionReadout position{0,"input",{
        {0,2.0,{{1,"first",.6},{2,"second",.3},{3,"third",.1}}},
        {1,1.0,{{2,"second",.7},{1,"first",.2}}},
        {3,.5,{{1,"first",.9}}}}};
    const auto ranks=rankCapacity(position);
    check(ranks==3);
    const auto paths=tokenTrajectories(report,position,VerticalAxis::Rank);
    check(paths.size()==4); // Two connected tokens, one singleton, one post-gap singleton.
    check(paths[0].tokenId==1 && paths[0].points.size()==2);
    check(paths[3].tokenId==1 && paths[3].points.size()==1);
    TokenTrajectory curve{1,{{-.9,.8,-1},{.8,-.9,-.5},{.85,-.9,0},{-.8,.9,.5},{-.8,0,1}}};
    for(size_t segment=0;segment+1<curve.points.size();++segment) {
        check(sampleTrajectory(curve,segment,0)==curve.points[segment]);
        check(sampleTrajectory(curve,segment,1)==curve.points[segment+1]);
        for(int step=0;step<=100;++step) {
            const double t=step/100.;
            const auto value=sampleTrajectory(curve,segment,t);
            for(size_t dim=0;dim<3;++dim) {
                check(std::isfinite(value[dim]));
                check(value[dim]>=std::min(curve.points[segment][dim],curve.points[segment+1][dim])-1e-12);
                check(value[dim]<=std::max(curve.points[segment][dim],curve.points[segment+1][dim])+1e-12);
            }
            check(near(value[2],curve.points[segment][2]+t*.5));
        }
    }
    // Adjacent spline segments meet with the same derivative, including extrema.
    for(size_t knot=1;knot+1<curve.points.size();++knot) {
        const double epsilon=1e-6;
        const auto left=sampleTrajectory(curve,knot-1,1-epsilon),right=sampleTrajectory(curve,knot,epsilon);
        for(size_t dim=0;dim<3;++dim)
            check(std::abs((curve.points[knot][dim]-left[dim])/epsilon-(right[dim]-curve.points[knot][dim])/epsilon)<2e-5);
    }
    const auto midpoint=sampleTrajectory(paths[0],0,.5);
    for(size_t dim=0;dim<3;++dim)check(near(midpoint[dim],(paths[0].points[0][dim]+paths[0].points[1][dim])*.5));
    auto dropout=position;
    dropout.layers[1].candidates.erase(dropout.layers[1].candidates.begin()+1);
    const auto broken=tokenTrajectories(report,dropout,VerticalAxis::Entropy);
    for(const auto& path:broken)if(path.tokenId==1)check(path.points.size()==1);
    const auto a=candidateLocation(report,position.layers[0],0,ranks,VerticalAxis::Rank);
    const auto b=candidateLocation(report,position.layers[0],1,ranks,VerticalAxis::Rank);
    const auto c=candidateLocation(report,position.layers[0],2,ranks,VerticalAxis::Rank);
    check(near(a[0],.2) && near(a[1],1) && near(a[2],-1));
    check(near(b[1],0) && near(c[1],-1));
    const auto smaller=candidateLocation(report,position.layers[1],1,ranks,VerticalAxis::Rank);
    check(near(smaller[1],b[1])); // Fewer candidates must not stretch a plane.
    check(near(candidateLocation(report,position.layers[2],0,ranks,VerticalAxis::Rank)[2],1));
    const auto entropy=candidateLocation(report,position.layers[0],0,ranks,VerticalAxis::Entropy);
    check(near(entropy[1],4/std::log(100.)-1));
    check(near(entropy[0],a[0]) && near(entropy[2],a[2]));
    auto delta=candidateChange(position,position.layers[1],0);
    check(delta.previousLayerAvailable && near(*delta.probabilityDelta,.4) && delta.rankGain==1);
    delta=candidateChange(position,position.layers[1],1);
    check(near(*delta.probabilityDelta,-.4) && delta.rankGain==-1);
    // No invented deltas across gaps, absent tokens, or unavailable readouts.
    check(!candidateChange(position,position.layers[0],0).probabilityDelta);
    check(!candidateChange(position,position.layers[2],0).previousLayerAvailable);
    position.layers[1].candidates[0].tokenId=4;
    delta=candidateChange(position,position.layers[1],0);
    check(delta.previousLayerAvailable && !delta.probabilityDelta && !delta.rankGain);
    position.layers[0].candidates.clear();
    check(!candidateChange(position,position.layers[1],0).previousLayerAvailable);
    // Single-layer, single-candidate and V=1 captures stay finite and centered.
    report.layerCount=1;report.vocabSize=1;
    LayerReadout single{0,0,{{0,"only",1.}}};
    const auto point=candidateLocation(report,single,0,1,VerticalAxis::Rank);
    check(near(point[0],1) && near(point[1],0) && near(point[2],0));
    check(std::isfinite(candidateLocation(report,single,0,1,VerticalAxis::Entropy)[1]));
    std::cout<<"Observatory geometry tests passed\n";
}
