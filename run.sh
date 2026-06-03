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

conda install -c conda-forge cudatoolkit=11.8.0 "cudnn<9.0" -y
conda install bioconda::chronumental

mkdir -p scripts/build
cd scripts/build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/g++" \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/gcc"
cmake --build . -j
cd ../../

python -m ipykernel install --user \
  --name ipynb-env \
  --display-name "Python (ipynb-env)"

jupyter lab --no-browser --port=8888


# https://github.com/TheSatoLab/CoVFit/blob/main/CoVFit_CLI/ReadMe.md
# chronumental --tree /home/pgangwar@AD.UCSD.EDU/WBE-LLM/SARS2-WBE/gisaidAndPublic.2025-09-01.masked.nwk.gz --reference_node 'CHN/Wuhan_IME-WH01/2019|MT291826.1|2019-12-30'  --dates /home/pgangwar@AD.UCSD.EDU/WBE-LLM/SARS2-WBE/gisaidAndPublic.2025-09-01.metadata.tsv.gz --only_use_full_dates --steps 2000 --use_gpu
# 1. Run get_amino_acid_mutations_of_sequences.py 
# 2. Run lineage_spike_from_mat.py
# 3. Run create_from_chronumental_data.ipynb
# 4. Run rank_analysis.ipynb
# 5. Run new_mutations_cumulative.ipynb
# 6. Run new_mutations_fitness.ipynb
# 7. Run scout_selection.ipynb


#python3 scripts/lineage_spike_from_mat.py --mat /data/sukhanna/MPP/gisaidAndPublic.2025-09-01.masked.pb.gz --reference ../WBE-LLM/SARS2-WBE/test/NC_045512v2.fa --lineages B,B.1.1.529,BA.1,BA.4,BA.5,XBB.1.5,JN.1 --output data/vaccine_recommendations.pkl