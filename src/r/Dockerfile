FROM rocker/r-ver:4.4.1

WORKDIR /app

# Install pinned R packages first (cached between builds)
COPY install_packages.R .
RUN Rscript install_packages.R

# Copy the script
COPY titanic_model.R .

# The dataset is NOT copied into the image; it is mounted at run time
ENV DATA_DIR=/app/data

CMD ["Rscript", "titanic_model.R"]