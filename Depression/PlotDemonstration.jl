using GLPK, Cbc, JuMP, SparseArrays
using Plots

df = Array(CSV.read("Depression/channel_data_interp.csv", DataFrame, header = 0))
