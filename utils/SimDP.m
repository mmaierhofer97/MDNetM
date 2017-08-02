function [sim,indices] = SimDP(start,conv,ovEnd,ovSamp,SampStart,SampEnd,x,ovF)
    for i = 1:length(x)
        frame_graph{1}{i}{1}=start{x(i)}{ovF(1)-1};
        frame_graph{1}{i}{2}=start{x(i)}{ovF(1)-1};
        Samp{1}{i}{1}=SampStart{x(i)}{ovF(1)-1};
        Samp{1}{i}{2}=SampStart{x(i)}{ovF(1)-1};
        for j = 1:length(ovF)
            frame_graph{j+1}{i}=conv{x(i)}{ovF(j)};
            Samp{j+1}{i}=ovSamp{x(i)}{ovF(j)};
            frame_graph{j+1}{i}{2}=frame_graph{j}{i}{2}(:,:,:,1);
            Samp{j+1}{i}{2}= Samp{j}{i}{2}(1,:);
        end
        frame_graph{length(ovF)+2}{i}{1}=ovEnd{x(i)};
        frame_graph{length(ovF)+2}{i}{2}=ovEnd{x(i)};
        Samp{length(ovF)+2}{i}{1}=SampEnd{x(i)};
        Samp{length(ovF)+2}{i}{2}=SampEnd{x(i)};
    end
    for i = 1:length(x)+1
        sim{1}(i)=1;
    end
    for i = 2:length(frame_graph)
         for j = 1:length(frame_graph{i})+1
             maxi=-1;
             index=-1;
             for k = 1:length(frame_graph{i-1})+1
                 s=0;
                 for l = 1:length(frame_graph{i})
                    s = s+mean([cosine_sim(frame_graph{i-1}{l}{(l~=k) +1},frame_graph{i}{l}{(l~=j)+1}),cosine_sim(frame_graph{1}{l}{(l~=k) +1},frame_graph{i}{l}{(l~=j)+1}),cosine_sim(frame_graph{end}{l}{(l~=k) +1},frame_graph{i}{l}{(l~=j)+1}),overlap_ratio(Samp{i-1}{l}{(l~=k) +1},Samp{i}{l}{(l~=j)+1})]);
                 end
                 s=s/length(frame_graph{i});
                 s=sim{i-1}(k)*s;
%                 s=max([cosine_sim(frame_graph{1}(:,:,:,1),frame_graph{i}(:,:,:,j)),cosine_sim(frame_graph{end}(:,:,:,1),frame_graph{i}(:,:,:,j))]);
                 if s>maxi
                     maxi = s;
                     index = k;
                 end    
             sim{i}(j)=maxi;
             ind{i}(j)= index;
             end
         end
    end
    indices=[];
    try indices(1)=ind{end}(1);
    end
    j=1;
    i=1;
    while (length(j)>0)&&length(ind)>2
        indices(end+1)=ind{end-i}(indices(end));
        j=ind{end-i-1};
        i=i+1;
    end
   indices = wrev(indices);

end
