# Txt tests in julia # 
using DelimitedFiles
f = readdlm("Depression/channel_data.txt",'\t',Float64,'\n')
println(f)
close(f)