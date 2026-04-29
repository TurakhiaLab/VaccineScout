conda create -n ipynb-env python=3.10
conda activate ipynb-env
conda install -y jupyterlab ipykernel
conda install numpy pandas matplotlib scipy
pip install seaborn scikit-learn
conda install -c conda-forge biopython pyarrow fastparquet tqdm pytorch umap-learn ipywidgets jupyterlab

python -m ipykernel install --user \
  --name ipynb-env \
  --display-name "Python (ipynb-env)"

jupyter lab --no-browser --port=8888