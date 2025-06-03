function dfs_topological_sort(adj_matrix::Array{Int, 2})
    num_vertices = size(adj_matrix, 1)
    visited = Set{Int}()
    stack = Vector{Int}()

    for i in 1:num_vertices
        if !(i in visited)
            dfs_topological_sort_helper(i, visited, stack, adj_matrix)
        end
    end

    return reverse(stack)

end 

function dfs_topological_sort_helper(v::Int, visited::Set{Int}, stack::Vector{Int}, adj_matrix::Array{Int, 2}) 
    push!(visited, v)

    for i in 1:size(adj_matrix, 1)
        if adj_matrix[v, i] == 1 && !(i in visited)
            dfs_topological_sort_helper(i, visited, stack, adj_matrix)
        end
    end

    push!(stack, v)
end

function khans_topological_sort(adj_matrix::Array{Int, 2})
    num_vertices = size(adj_matrix, 1)
    topo = Vector{Int}()
    s = Set{Int}()

    for i in 1:num_vertices
        include = true
        for j in 1:num_vertices
            if adj_matrix[j, i] == 1
                include = false
                break
            end
        end
        if include 
            push!(s, i)
        end
    end
    
    while !isempty(s)
        curr = pop!(s)
        push!(topo, curr)
        for i in 1:num_vertices
            if adj_matrix[curr, i] == 1
                adj_matrix[curr, i] = 0
                has_edge = false
                for j in 1:num_vertices
                    if adj_matrix[j, i] == 1
                        has_edge = true
                        break
                    end
                end
                if !has_edge 
                    push!(s, i)
                end
            end
        end
    end
    
    return topo
end

const V = 6
adj_matrix = zeros(Int, V, V)

adj_matrix[3, 4] = 1
adj_matrix[4, 2] = 1
adj_matrix[5, 1] = 1
adj_matrix[5, 2] = 1
adj_matrix[6, 1] = 1
adj_matrix[6, 3] = 1

println("DFS Topological Sort: ", dfs_topological_sort(adj_matrix))
println("Kahn's Topological Sort: ", khans_topological_sort(adj_matrix))
