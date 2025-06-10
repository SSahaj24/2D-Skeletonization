function linePath = getArcsProperties_Skel(node_graph, maskedimage)
G = node_graph;
node_degree = degree(node_graph);
branchPointsIdx = find(node_degree>2);
endPointsIdx= find(node_degree==1);
pointsIdxN = [endPointsIdx; branchPointsIdx];
nodesPixelIndex = node_graph.Nodes.PixelIndex(:);
fid = fopen("paths.txt", "w");
if ~isempty(pointsIdxN)
    startNode = pointsIdxN(1);
else
    startNode= 1;
end

%% Traversal Initialization

currentNode = startNode;
if numedges(G)
    nbr = neighbors(G, currentNode);
    neighbor = nbr(1);
    fprintf(fid, "%d ", currentNode);
    edgeIndex = findedge(G, currentNode, neighbor);
    G = rmedge(G, edgeIndex);
    currentNode = neighbor;
end

ndT = node_degree;

branchStack = [];
branchNbrStack = [];
while(numedges(G))
    % disp(currentNode);
    % if currentNode == 543
    %     pause;
    % end
    % [~, idxNZ] = find(ndT==0);
    % G = rmnode(G,idxNZ);
    if ~ismember(currentNode,pointsIdxN)
        if(length(neighbors(G, currentNode)))
            fprintf(fid, "%d ", currentNode);
            nbr = neighbors(G, currentNode);
            neighbor=nbr(1);
            edgeIndex = findedge(G, currentNode, neighbor);
            G = rmedge(G, edgeIndex);
            currentNode = neighbor;
        else
            fprintf(fid, "%d\n", currentNode);
            if ~isempty(branchNbrStack)
                idxB = find(G.Nodes.PixelIndex==branchStack(1));
                idxN = find(G.Nodes.PixelIndex==branchNbrStack(1));
                branchStack = branchStack(2:end);
                branchNbrStack = branchNbrStack(2:end);
                edgeIndex = findedge(G, idxN, idxB);
                while ~edgeIndex
                    if isempty(branchNbrStack)
                        currentNode = G.Edges.EndNodes(1);
                        break;
                    end
                    idxB = find(G.Nodes.PixelIndex==branchStack(1));
                    idxN = find(G.Nodes.PixelIndex==branchNbrStack(1));
                    branchStack = branchStack(2:end);
                    branchNbrStack = branchNbrStack(2:end);
                    edgeIndex = findedge(G, idxN, idxB);

                    currentNode = idxN;
                    % fprintf(fid, "%d ", idxB);
                    % G = rmedge(G, edgeIndex);

                end
                if edgeIndex
                    currentNode = idxN;
                    fprintf(fid, "%d ", idxB);
                    G = rmedge(G, edgeIndex);
                end
            else
                currentNode = G.Edges.EndNodes(1);
            end
        end

        %% If it is a branchpoint
    elseif ismember(currentNode, branchPointsIdx)
        fprintf(fid, "%d\n", currentNode);
        nbr = neighbors(G, currentNode);
        if ~isempty(nbr)
            for iB = 2 : length(nbr)
                branchStack= [nodesPixelIndex(currentNode); branchStack];
            end
            for iB = 2 : length(nbr)
                branchNbrStack= [nodesPixelIndex(nbr(iB)); branchNbrStack];
            end
            neighbor = nbr(1);
            ndT(currentNode) = ndT(currentNode)-1;
            ndT(neighbor) = ndT(neighbor)-1;
            edgeIndex = findedge(G, currentNode, neighbor);
            G = rmedge(G, edgeIndex);
            currentNode = neighbor;
        else
            fprintf(fid, "%d\n", currentNode);
            if ~isempty(branchNbrStack)
                idxB = find(G.Nodes.PixelIndex==branchStack(1));
                idxN = find(G.Nodes.PixelIndex==branchNbrStack(1));
                branchStack = branchStack(2:end);
                branchNbrStack = branchNbrStack(2:end);
                edgeIndex = findedge(G, idxN, idxB);

                while ~edgeIndex
                    if isempty(branchNbrStack)
                        currentNode = G.Edges.EndNodes(1);
                        break;
                    end
                    idxB = find(G.Nodes.PixelIndex==branchStack(1));
                    idxN = find(G.Nodes.PixelIndex==branchNbrStack(1));
                    branchStack = branchStack(2:end);
                    branchNbrStack = branchNbrStack(2:end);
                    edgeIndex = findedge(G, idxN, idxB);

                    currentNode = idxN;
                    % fprintf(fid, "%d ", idxB);
                    % G = rmedge(G, edgeIndex);

                end
                if edgeIndex
                    currentNode = idxN;
                    fprintf(fid, "%d ", idxB);
                    G = rmedge(G, edgeIndex);
                end
            else
                currentNode = G.Edges.EndNodes(1);
            end
        end


        %% If it is an endpoint
    elseif ismember(currentNode,endPointsIdx)
        fprintf(fid, "%d\n", currentNode);
        if ~isempty(branchNbrStack)
            idxB = find(G.Nodes.PixelIndex==branchStack(1));
            idxN = find(G.Nodes.PixelIndex==branchNbrStack(1));
            branchStack = branchStack(2:end);
            branchNbrStack = branchNbrStack(2:end);
            edgeIndex = findedge(G, idxN, idxB);
            while ~edgeIndex
                if isempty(branchNbrStack)
                    currentNode = G.Edges.EndNodes(1);
                    break;
                end
                idxB = find(G.Nodes.PixelIndex==branchStack(1));
                idxN = find(G.Nodes.PixelIndex==branchNbrStack(1));
                branchStack = branchStack(2:end);
                branchNbrStack = branchNbrStack(2:end);
                edgeIndex = findedge(G, idxN, idxB);

                currentNode = idxN;
                % fprintf(fid, "%d ", idxB);
                % G = rmedge(G, edgeIndex);

            end
            if edgeIndex
                currentNode = idxN;
                fprintf(fid, "%d ", idxB);
                G = rmedge(G, edgeIndex);
            end
        else
            currentNode = G.Edges.EndNodes(1);
        end
    end
end
fprintf(fid, "%d", currentNode);
fclose(fid);

%% Create Data Structure for each line arc
L = readlines("paths.txt");
cnt = 1;

for i = 1 : length(L)
    % disp(i);
    str = split(L(i), ' ');
    vals = [];
    for j = 1 : length(str)
        vals(j) = str2num(str(j));
    end
    if length(vals)>2
        if ~ismember(vals(1),branchPointsIdx)
            nbr = neighbors(node_graph, vals(1));
            for iN = length(nbr)
                if ismember(nbr(iN), branchPointsIdx)
                    vals = [nbr(iN) vals];
                end
            end
        end
        if ~ismember(vals(end),branchPointsIdx)
            nbr = neighbors(node_graph, vals(end));
            for iN = length(nbr)
                if ismember(nbr(iN), branchPointsIdx)
                    vals = [vals nbr(iN)];
                end
            end
        end

        linePath(cnt).arc = nodesPixelIndex(vals);
        [linePath(cnt).x, linePath(cnt).y] = ind2sub(size(maskedimage), ...
            linePath(cnt).arc);
        linePath(cnt).Pos = [linePath(cnt).x linePath(cnt).y];
        %% Find line Lengths
        dX = diff(linePath(cnt).x);
        dY = diff(linePath(cnt).y);
        segment_lengths = sqrt(dX.^2 + dY.^2);
        linePath(cnt).length = sum(segment_lengths);

        cnt = cnt + 1;
    end
end
if ~exist("linePath","var")
        linePath.length = 0;
end
end



