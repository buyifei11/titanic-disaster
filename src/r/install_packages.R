# Install pinned versions of the R packages used by titanic_model.R
install.packages("remotes")

remotes::install_version("readr", version = "2.1.5", upgrade = "never")
remotes::install_version("dplyr", version = "1.1.4", upgrade = "never")

for (p in c("readr", "dplyr")) {
  if (!requireNamespace(p, quietly = TRUE)) stop(paste("Failed to install", p))
  cat("Installed", p, as.character(packageVersion(p)), "\n")
}