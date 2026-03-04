### Setting libraries ###
library(xvm) # This library is the one that reads .xpm and .xvg files. The documentation can be found at https://cran.r-project.org/web/packages/xvm/xvm.pdf 

### Setting directories
xpm_dir <- "/Users/lorenzosisti/Downloads/Gibbs/"

# Now let's set the paths of the .xpm gibbs energy and of the firstplane (where we added the frame information) 
gibbs_xpm_path <- paste(xpm_dir, "gibbs.xpm", sep = "") # Calcolato con gmx sham
time_xvg_path <- paste(xpm_dir, "firstplane_with_time.xvg", sep = "")

# Read and plot gibbs energy
xpm_data <- read_xpm(gibbs_xpm_path)
plot_xpm(xpm_data)
plot_xpm_3d(xpm_data)

# Let's now keep just the rows of the gibbs energy dataframe corresponding to the energy minima
pc_df <- xpm_data$gibbs.xpm$data # Select just the information of interests
pc_coords <- pc_df[, c("value", "x_actual", "y_actual"), drop = FALSE]
pc_zero <- subset(pc_coords, value == 0)

# Read and parse the .xvg file containing the first two eigenvectors and the timeframe corresponding to them
xvg_data <- read_xvg(time_xvg_path)
pc_time_df <- xvg_data$firstplane_with_time.xvg$data

# Define a function that finds a small volume around the minima, so that we can find (from the .xvg file) the frame contained in the minima volumes
match_with_tol <- function(df_time, df_zero, tol) {
  keep <- logical(nrow(df_time))
  for (i in seq_len(nrow(df_zero))) {
    dx <- abs(df_time$Y_1 - df_zero$x_actual[i])
    dy <- abs(df_time$Y_2 - df_zero$y_actual[i])
    keep <- keep | (sqrt(dx^2 + dy^2) < tol)
  }
  df_time[keep, ]
}

# Execute the function
matched_df <- match_with_tol(pc_time_df, pc_zero, tol = 0.1)
