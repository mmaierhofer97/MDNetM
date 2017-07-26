function [sim,indices] = SimDP(start,conv,x)
    start=start{x};
    x
    conv
    conv=conv{x}
    frame_graph{1}=start;
    for i = 1:length(conv)
        frame_graph{i+1}=conv{i};
    end
    frame_graph{length(iFrames)+2}=last;
    
    sim{1}=1;

    for i = 2:length(frame_graph)
        for j = 1:length(frame_graph{i}(1,1,1,:))
            maxi=-1;
            index=-1;
            for k = 1:length(frame_graph{i-1}(1,1,1,:))
                %s=sim{i-1}(k)*
                s=max([cosine_sim(frame_graph{1}(:,:,:,1),frame_graph{i}(:,:,:,j)),cosine_sim(frame_graph{end}(:,:,:,1),frame_graph{i}(:,:,:,j))]);
                if s>maxi
                    maxi = s;
                    index = k;
                end    
            sim{i}(j)=maxi;
            [ind{i}(j)]=index;
            end
        end
    end
    indices=[];
    indices(1)=ind{end};
    j=1;
    i=1;
    while (length(j)>0)
        indices(end+1)=ind{end-i}(indices(end));
        j=ind{end-i-1};
        i=i+1;
    end
   indices = wrev(indices);

end
