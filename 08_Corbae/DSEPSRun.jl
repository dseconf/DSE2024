############################################################################################################
# Simulated Method of Moments 
# DSE Summer School 2024
# Professor: Dean Corbae
############################################################################################################

using Plots, Tables, DataFrames, CSV

include("Model.jl");

D_seed = 1500;
R_seed = 200;

true_data = Initialize_True_Data(; seed = D_seed);

# Plot true data
plot(true_data.x₀, legend = false, ylabel = L"x_{t}", xlabel= L"t", color = "black")
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/AR1.pdf")

# Main: Using different moments
@elapsed res_4 = estimate(1:2; D_seed = D_seed, R_seed = R_seed)
@elapsed res_5 = estimate(2:3; D_seed = D_seed, R_seed = R_seed)
@elapsed res_6 = estimate(1:3; D_seed = D_seed, R_seed = R_seed)

# J_TH surface for first-stage (i.e., using 𝕎 = I)
plot_3d_J_TH(res_4, true_data, 1)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_1_1st.pdf")
plot_3d_J_TH(res_5, true_data, 1)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_2_1st.pdf")
plot_3d_J_TH(res_6, true_data, 1)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_3_1st.pdf")

# J_TH surface for second-stage (i.e., using 𝕎 = S_TH)
plot_3d_J_TH(res_4, true_data, 2)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_1_2st.pdf")
plot_3d_J_TH(res_5, true_data, 2)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_2_2st.pdf")
plot_3d_J_TH(res_6, true_data, 2)
Plots.savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/3D_3_2st.pdf")

# Tables

#𝐽 matrix of derivatives
CSV.write("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Tables/jacobian_1_4.csv", Tables.table(res_4.𝐽₁));
CSV.write("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Tables/jacobian_2_4.csv", Tables.table(res_4.𝐽₂));
CSV.write("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Tables/jacobian_1_5.csv", Tables.table(res_5.𝐽₁));
CSV.write("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Tables/jacobian_2_5.csv", Tables.table(res_5.𝐽₂));
CSV.write("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Tables/jacobian_1_6.csv", Tables.table(res_6.𝐽₁));
CSV.write("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Tables/jacobian_2_6.csv", Tables.table(res_6.𝐽₂));

# Summary table
summ_table = create_table([res_4, res_5, res_6])
CSV.write("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Tables/summary.csv", summ_table);

# Bootstrapping
res_6_bootstrap = bootstrap_se(1:3)

histogram(res_6_bootstrap[:, 1], fillalpha=0.5, label = L"\hat{\rho}_{1}", c = :blue);
histogram!(res_6_bootstrap[:, 2], fillalpha=0.5, label = L"\hat{\rho}_{2}", c = :red)
savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/Histogram_rho.pdf")

histogram(res_6_bootstrap[:, 3], fillalpha=0.5, label = L"\hat{\sigma}_{1}", c = :blue);
histogram!(res_6_bootstrap[:, 4], fillalpha=0.5, label = L"\hat{\sigma}_{2}",c = :red)

savefig("/Users/smlm/Desktop/Desktop - Stefano’s MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS7-Solution/Figures/Histogram_sigma.pdf")
