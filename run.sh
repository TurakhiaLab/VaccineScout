conda create -n ipynb-env python=3.10
conda activate ipynb-env
conda install -y jupyterlab ipykernel
conda install numpy pandas matplotlib scipy
pip install seaborn scikit-learn
conda install -c conda-forge biopython pyarrow fastparquet tqdm pytorch umap-learn ipywidgets jupyterlab
conda install -c conda-forge "arrow-cpp>=19.0.1"

conda install -c conda-forge --solver=libmamba "libprotobuf=4.25.3"
conda install -c conda-forge --solver=libmamba "libboost-devel=1.85"
conda install -c bioconda -c conda-forge bte --solver=libmamba

conda install -c conda-forge tbb tbb-devel --solver=libmamba
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

bash scripts/fetch_usher.sh

mkdir -p scripts/build
cd scripts/build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/g++" \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/gcc" \
  -DBUILD_GET_MUTATIONS=ON
cmake --build . -j
cd ../../

python -m ipykernel install --user \
  --name ipynb-env \
  --display-name "Python (ipynb-env)"

jupyter lab --no-browser --port=8888



# 1. Run get_amino_acid_mutations_of_sequences.py 
# 2. Run create_from_chronumental_data.ipynb
# 3. Run new_mutations_cumulative.ipynb
# 4. 