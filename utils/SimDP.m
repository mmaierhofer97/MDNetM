function [sim,indices] = SimDP(start,conv,ovEnd,x)
    for i = 1:length(x)
        frame_graph{1}{i}{1}=start{x(i)};
        frame_graph{1}{i}{2}=start{x(i)};
        for j = 1:length(conv{i})
            frame_graph{j+1}{i}=conv{x(i)}{j};
        end
        frame_graph{length(conv{i})+1}{i}{1}=ovEnd{x(i)};
        frame_graph{length(conv{i})+1}{i}{2}=ovEnd{x(i)};
    end
    for i = 1:length(x)
        sim{1}(i)=1;
    end
    for i = 2:length(frame_graph)
         for j = 1:length(frame_graph{i})
             maxi=-1;
             index=-1;
             for k = 1:length(frame_graph{i-1})
                 s=0;
                 for l = 1:length(frame_graph{i})
                    s = s+cosine_sim(frame_graph{i-1}{l}{(l~=k) +1},frame_graph{i}{j}{(l~=j)+1});
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
    indices(1)=ind{end}(1);
    j=1;
    i=1;
    while (length(j)>0)&&length(ind)>2
        indices(end+1)=ind{end-i}(indices(end));
        j=ind{end-i-1};
        i=i+1;
    end
   indices = wrev(indices);

end
