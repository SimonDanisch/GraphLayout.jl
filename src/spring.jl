"""
    Use the spring/repulsion model of Fruchterman and Reingold (1991):
        Attractive force:  f_a(d) =  d^2 / k
        Repulsive force:  f_r(d) = -k^2 / d
    where d is distance between two vertices and the optimal distance
    between vertices k is defined as C * sqrt( area / num_vertices )
    where C is a parameter we can adjust
    Arguments:
    adj_matrix Adjacency matrix of some type. Non-zero of the eltype
               of the matrix is used to determine if a link exists,
               but currently no sense of magnitude
    C          Constant to fiddle with density of resulting layout
    MAXITER    Number of iterations we apply the forces
    INITTEMP   Initial "temperature", controls movement per iteration
"""
function layout_spring{T}(adj_matrix::Array{T,2}; C=2.0, MAXITER=100, INITTEMP=2.0)
    N = size(adj_matrix, 1)
    # Initial layout is random on the square [-1,+1]^2
    locs = 2*rand(Point{2, Float64}, N) .- 1.
    layout_spring!(adj_matrix, locs, C=C, MAXITER=MAXITER, INITTEMP=INITTEMP)
end
scaler(z, a, b) = 2.0*((z - a)./(b - a)) - 1.0

immutable SpringLayouter{A, P, F}
    adjacency::A
    positions::P
    forces::F
    K::Float64
    initial_temperature::Float64
end

function iterate!{A,T,F}(sl::SpringLayouter{A,T,F}, state::Integer)
    positions = sl.positions;forces = sl.forces;adj_matrix = sl.adjacency
    const N = length(positions)
    const adj_zero = zero(eltype(A))
    const P = eltype(T)
    const D = length(P)
    P = eltype(T);PT = eltype(P);FT = eltype(F)
    K = sl.K
    # Calculate forces
    @inbounds for i = 1:N
        force_vec = FT(0)
        for j = 1:N
            i == j && continue
            d_vec = FT(positions[j] - positions[i])
            d = norm(d_vec)
            if ((adj_matrix[i,j] != adj_zero) || (adj_matrix[j,i] != adj_zero))
                F_d = PT(d / K - K^2 / d^2)
            else
                F_d = PT(-K^2 / d^2)
            end
            force_vec += FT(F_d.*d_vec)
        end
        forces[i] = force_vec
    end

    temperature = PT(sl.initial_temperature / state)
    # Now apply them, but limit to temperature
    @inbounds for i = 1:N
        f = forces[i]
        force_mag  = norm(f)
        scale      = min(force_mag, temperature)/force_mag
        positions[i] += P(f * scale)
    end
    nothing
end

function SpringLayouter(adj_matrix, initial_positions, C, initial_temperature)
    size(adj_matrix, 1) != size(adj_matrix, 2) && error("Adj. matrix must be square.")
    N = size(adj_matrix, 1)
    N != length(initial_positions) && error("length of initial positions must be equal to number of graph nodes.")
    # The optimal distance bewteen vertices
    K = C * sqrt(4.0 / N)
    # Store forces and apply at end of iteration all at once
    P = eltype(initial_positions)
    forces = zeros(Vec{length(P), eltype(P)}, N)
    SpringLayouter(adj_matrix, initial_positions, forces, K, initial_temperature)
end

function layout_spring!{T<:AbstractVector}(adj_matrix, initial_positions::T; C=2.0, MAXITER=100, INITTEMP=2.0)
    sl = SpringLayouter(adj_matrix, initial_positions, C, INITTEMP)
    # Iterate MAXITER times
    for iter = 1:MAXITER
        iterate!(sl, iter)
    end

    return sl.positions::T
end
