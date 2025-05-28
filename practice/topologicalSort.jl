const V = 6
adj_matrix = zeros(Int, V, V)

adj_matrix[3, 4] = 1
adj_matrix[4, 2] = 1
adj_matrix[5, 1] = 1
adj_matrix[5, 2] = 1
adj_matrix[6, 1] = 1
adj_matrix[6, 3] = 1

function topological_sort(adj_matrix::Array{Int, 2})
    num_vertices = size(adj_matrix, 1)
    visited = Set{Int}()
    stack = Vector{Int}()

    for i in 1:num_vertices
        if !(i in visited)
            topologicalSortUtil(i, visited, stack, adj_matrix)
        end
    end

    return reverse(stack)

end 

function topologicalSortUtil(v::Int, visited::Set{Int}, stack::Vector{Int}, adj_matrix::Array{Int, 2}) 
    push!(visited, v)

    for i in 1:size(adj_matrix, 1)
        if adj_matrix[v, i] == 1 && !(i in visited)
            topologicalSortUtil(i, visited, stack, adj_matrix)
        end
    end

    push!(stack, v)
end

println("Topological Sort: ", topological_sort(adj_matrix))