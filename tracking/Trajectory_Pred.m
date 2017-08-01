function [ resultLoc ] = Trajectory_Pred(prevLocs,sucFrames,curFrame)
    x=sucFrames(curFrame-sucFrames<6);
    prevLocs=squeeze(prevLocs);
    if length(x)>2
        y=prevLocs(x,:);
        p1=polyfit(x,y(:,1)',1);
        p2=polyfit(x,y(:,2)',1);
        p3=polyfit(x,y(:,3)',1);
        p4=polyfit(x,y(:,4)',1);
        resultLoc=[p1(1)*curFrame+p1(2),p2(1)*curFrame+p2(2),max(p3(1)*curFrame+p3(2),1),max(p4(1)*curFrame+p4(2),1)];
    else
        resultLoc=prevLocs(curFrame-1,:);

end