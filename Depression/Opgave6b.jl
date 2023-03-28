# Txt tests in julia # 

using GLPK, Cbc, JuMP, SparseArrays
using CSV
using DataFrames
using DelimitedFiles
using Plots



function constructA(H,K)
    h = length(H)
    k = length(K)-1
    A = zeros(h,h+2*k)

    for i in 1:h
        A[i,i:i+2*k] = hcat(transpose(reverse(K[2:end])), K)
    end
    
    A = A[:,k+1:end-k]
    B = H .+ 10

    # Make a function that returns A when given H and K
    return A, B
end

K = [
300 140 40
]

K = [[300 140 40], [ 500 230 60], [1000 400 70 ]]


df = Array(CSV.read("Depression/channel_data_interp.csv", DataFrame, header = 0))

function compute_time_Cbc(n, n_cores)
    H = df[1:n,2]
    h = length(H)

    A0, B = constructA(H,K[1])
    A1, _ = constructA(H,K[2])
    A2, _ = constructA(H,K[3])

    myModel = Model(Cbc.Optimizer)
        # If your want ot use GLPK instead use:
    # myModel = Model(GLPK.Optimizer)

    @variable(myModel, x0[1:h], Bin )
    @variable(myModel, x1[1:h], Bin )
    @variable(myModel, x2[1:h], Bin )
    @variable(myModel, R[1:h] >= 0 )
    @variable(myModel, t[1:h] >= 0)
    @variable(myModel, y[1:h-1], Bin)


    @objective(myModel, Min, sum(t))

    @constraint(myModel, y .== x0[1:h-1] .+ x0[2:h] .+ x1[1:h-1] .+ x1[2:h] .+ x2[1:h-1] .+ x2[2:h])
    @constraint(myModel, x0 .+ x1 .+ x2 .<= 1)
    @constraint(myModel, t .>= R .- B)
    @constraint(myModel, t .>= B .- R)
    @constraint(myModel, R .>= B )
    @constraint(myModel, [i=1:h],R[i] == sum(A0[i,j]*x0[j] .+ A1[i,j]*x1[j] .+ A2[i,j]*x2[j] for j=1:h))

    # set_optimizer_attribute(myModel, "maxNodes", 1000000)
    set_optimizer_attribute(myModel, "threads", n_cores)
    set_optimizer_attribute(myModel, "logLevel", 0)

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL || termination_status(myModel) == MOI.NODE_LIMIT
        println(" Done!")
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end

    tt = JuMP.solve_time(myModel)
    objective = JuMP.objective_value(myModel)
    nbombs = sum(JuMP.value.(x0) .+ JuMP.value.(x1) .+ JuMP.value.(x2))

    return tt, objective, nbombs
end

function compute_time_GLPK(n)
    H = df[1:n,2]
    h = length(H)

    A0, B = constructA(H,K[1])
    A1, _ = constructA(H,K[2])
    A2, _ = constructA(H,K[3])

    # myModel = Model(Cbc.Optimizer)
        # If your want ot use GLPK instead use:
    myModel = Model(GLPK.Optimizer)


    @variable(myModel, x0[1:h], Bin )
    @variable(myModel, x1[1:h], Bin )
    @variable(myModel, x2[1:h], Bin )
    @variable(myModel, R[1:h] .>= 0 )
    @variable(myModel, t[1:h] .>= 0)
    @variable(myModel, y[1:h-1], Bin)


    @objective(myModel, Min, sum(t))

    @constraint(myModel, y .== x0[1:h-1] .+ x0[2:h] .+ x1[1:h-1] .+ x1[2:h] .+ x2[1:h-1] .+ x2[2:h])
    # @constraint(myModel, x0 .+ x1 .+ x2 .<= 1)
    @constraint(myModel, t .>= R .- B)
    @constraint(myModel, t .>= B .- R)
    @constraint(myModel, R .>= B )
    @constraint(myModel, [i=1:h],R[i] == sum(A0[i,j]*x0[j] .+ A1[i,j]*x1[j] .+ A2[i,j]*x2[j] for j=1:h))

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL || termination_status(myModel) == MOI.NODE_LIMIT
        println(" Done!")
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end

    tt = JuMP.solve_time(myModel)
    objective = JuMP.objective_value(myModel)
    nbombs = sum(JuMP.value.(x0) .+ JuMP.value.(x1) .+ JuMP.value.(x2))

    return tt, objective, nbombs
end


# nR =  JuMP.value.(R)

m = 33
scale = 1
n_cores = 16

total_time_cbc_16 = zeros(m)
total_time_cbc_1 = zeros(m)
total_time_glpk = zeros(m)
objective_value = zeros(m)
number_bombs = zeros(m)

for i in 1:m
    n = scale*i
    print("Cbc16,i = ", i)
    t, obj, nb = compute_time_Cbc(n, n_cores)
    total_time_cbc_16[i] = t
    print("Cbc1,i = ", i)
    t, obj, nb = compute_time_Cbc(n, 1)
    total_time_cbc_1[i] = t
    print("GLPK, i = ", i)
    t, obj, nb = compute_time_GLPK(n)
    total_time_glpk[i] = t
    objective_value[i] = obj
    number_bombs[i] = nb
end

# println(total_time)
# println(objective_value)
# println(number_bombs)

plot((1:m)*scale,total_time_cbc_16, yaxis=:log, lw = 2, xlabel = "Number of Data Points", ylabel = "Computation Time [s]", yticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], title = "Computational Scaling", label = "Cbc multi-thread", legend = :bottomright)
plot!((1:m)*scale,total_time_cbc_1, lw = 2, label = "Cbc single thread")
plot!((1:m)*scale,total_time_glpk, lw = 2, label = "GLPK")