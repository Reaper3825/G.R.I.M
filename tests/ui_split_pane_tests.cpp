#include "ui/primitives/ui_split_pane.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/input/ui_input_router.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

void runSplitPaneTests() {
    auto check=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    auto closeEnough=[](float a,float b){return std::abs(a-b)<0.01f;};
    auto& router=GRIM::UI::InputRouter::get();
    UISplitPane split(UISplitPane::Axis::Horizontal);
    split.setPosition(10,20);split.setSize(800,400);split.setMinimumSizes(160,200);split.setDefaultSecondSize(268);
    int clicks=0;UIButton sibling("Sibling",[&]{++clicks;});sibling.setPosition(0,0);sibling.setSize(1000,1000);
    OverlayRenderer sink;
    auto frame=[&](InputState in,bool cancel=false){router.beginFrame(cancel);router.currentPanel=93;split.update(in,0);sibling.update(in,0);split.drawOverlay(sink,{});router.endFrame();};
    check(closeEnough(split.firstRect().size.x,520) && closeEnough(split.secondRect().origin.x,542),"Split defaults do not partition parent");
    InputState in;in.mousePos={536,50};in.mousePressed[0]=in.mouseDown[0]=true;frame(in);
    check(split.isResizing() && split.hasPointerCapture(),"Divider did not capture");
    in.mousePressed[0]=false;in.mousePos.x=436;frame(in);
    check(closeEnough(split.secondRect().size.x,368),"Divider drag did not resize pane");
    in.mouseDown[0]=false;in.mouseReleased[0]=true;frame(in);
    check(!split.hasPointerCapture() && clicks==0,"Divider release leaked to sibling");
    split.setSize(1000,400);check(closeEnough(split.secondRect().size.x,368),"Outer resize lost preferred pane size");
    split.setSize(100,400);
    check(closeEnough(split.firstRect().size.x+split.dividerRect().size.x+split.secondRect().size.x,100),"Tiny split overlaps or exceeds parent");
    split.setSize(800,400);check(closeEnough(split.secondRect().size.x,368),"Tiny parent destroyed preferred size");
    auto press=[&]{in={};const auto d=split.dividerRect();in.mousePos={d.origin.x+6,50};in.mousePressed[0]=in.mouseDown[0]=true;frame(in);in.mousePressed[0]=false;};
    press();in.mousePos.x=-100;frame(in);
    check(closeEnough(split.firstRect().size.x,160),"First pane minimum ignored");
    frame(in,true);check(closeEnough(split.secondRect().size.x,368) && !split.isResizing(),"Cancellation did not restore split");
    press();in.mousePos.x=2000;frame(in);check(closeEnough(split.secondRect().size.x,200),"Second pane minimum ignored");
    split.setVisible(false);check(!split.hasPointerCapture() && closeEnough(split.secondRect().size.x,368),"Hide did not cancel resize");
    split.setVisible(true);in={};in.mousePos={split.dividerRect().origin.x+6,50};in.mousePressed[1]=true;frame(in);
    check(!split.hasCustomSize() && closeEnough(split.secondRect().size.x,268),"Divider reset failed");
    split.setSecondSize(std::numeric_limits<float>::quiet_NaN());check(!split.hasCustomSize(),"Nonfinite size accepted");
    in={};in.mousePos={536,50};in.mousePressed[0]=in.mouseDown[0]=true;
    frame(in.clipped({0,0},{20,20}));check(!split.isResizing(),"Divider ignored ancestor clip");
    UISplitPane vertical(UISplitPane::Axis::Vertical);vertical.setPosition(10,20);vertical.setSize(800,600);
    vertical.setMinimumSizes(100,130);vertical.setSecondSize(210);
    check(closeEnough(vertical.secondRect().origin.y,410) && closeEnough(vertical.firstRect().size.y,378),"Vertical split geometry incorrect");
    check(closeEnough(vertical.secondRect().size.x,800),"Vertical split changed cross axis");
    std::cout<<"Split pane geometry, capture, minimums, cancellation and reset tests passed\n";
}
