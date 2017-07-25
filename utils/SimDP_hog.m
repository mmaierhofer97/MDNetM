function [sim,indices] = SimDP_hog(img,samp)
    frame_graph{1}{1}=imcrop(img{1},samp{1});
    
    for i = 2:length(img)-1
        crop=[];
        for j = 1:length(samp{i})
            crop{j} = imcrop(img{i},samp{i}(j,:));
        end
        frame_graph{i} = crop;
    end
    
    frame_graph{length(img)}{1} = imcrop(img{end},samp{end});
    
    sim{1}=1;
    indices=[1];
    for i = 2:length(frame_graph)
        i
        for j = 1:length(frame_graph{i})
            maxi=-1;
            index=-1;
            for k = 1:length(frame_graph{1})
                %s=sim{i-1}(k)*
                s=max([hog_sim(frame_graph{1}{1},frame_graph{i}{j}),hog_sim(frame_graph{end}{1},frame_graph{i}{j})]);
                if s>maxi
                    maxi = s;
                    index = k;
                end
            end
            sim{i}(j)=maxi;
        end
        [x,indices(i)]=max(sim{i});
    end
    indices=indices(1:end-1);
end
