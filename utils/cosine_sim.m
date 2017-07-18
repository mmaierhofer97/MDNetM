function [sim] = cosine_sim(imA,imB)
    
    
    sim=DDot(imA,imB)/sqrt(DDot(imA,imA)*DDot(imB,imB));

    function [res] = DDot(A,B)
         res = sum(sum(dot(A,B)));
    end

end
