addpath(genpath("D:\school\y4s1\fyp2\neural-network-for-nonlocality-in-networks\matlab\QETLAB"))
addpath(genpath("D:\school\y4s1\fyp2\neural-network-for-nonlocality-in-networks\matlab\cvx"))





wid = fopen("3333seqmat.txt", "w")

fid = fopen("3333seq.txt", "rt");
while true
    line = fgetl(fid);
    if ~ischar(line); break; end
    tokens = split(line);
    tokens(1)
    amt = get_threshhold(3,3,tokens(1));
    fprintf(wid, '%s %.4f\n', convertCharsToStrings(tokens(1)), amt);
end
fclose(fid);
fclose(wid);
function noisethres = get_threshhold(alice_in, bob_in, str)
    desc = [3 3 alice_in bob_in];
    prbox = get_peer_box(alice_in, bob_in, str);

    p_random = get_noise(alice_in, bob_in);
    cvx_begin
        variable lambda;
        minimize lambda;
        subject to
            NPAHierarchy(lambda*p_random + (1-lambda)*prbox,desc,2) == 1;
   cvx_end
   noisethres = cvx_optval;
end

function noise = get_noise(alice_in, bob_in)
    dim1 = alice_in*2 + 1;
    dim2 = bob_in*2+1;
    box = ones(dim1, dim2);
    box(1,1) = 9;
    box(1,2:end) = box(1,2:end) + 2 ;
    box(2:end,1) = box(2:end,1) + 2 ;
    noise = box/9;
end
function pr = get_peer_box(alice_in, bob_in, str)
    p1 = [1,0; 0,1];
    p2 = [1,0; 0,0];
    p3 = [0,1; 1,0];
    p4 = [0,1; 0,0];
    p5 = [0,0; 1,0];
    p6 = [0,0; 0,1];
    prr = cat(3,p1,p2,p3,p4,p5,p6);
    arr = str{1};
    dim1 = alice_in*2 + 1;
    dim2 = bob_in*2+1;
    box = zeros(dim1, dim2);
    box(1,1) = 3;
    box(1,2:end) = box(1,2:end) + 1 ;
    box(2:end,1) = box(2:end,1) + 1 ;
    for a = 1:alice_in
        for b = 1:bob_in
            index = (a-1)*bob_in + b;
            box(2*a:2*a+1, 2*b:2*b+1) = prr(:,:,str2double(arr(index:index)) + 1);
        end
    end
    pr = box/3;
end
