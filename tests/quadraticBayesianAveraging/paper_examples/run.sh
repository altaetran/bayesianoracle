cd ../../../
python setup.py build
python setup.py install
cd tests/quadraticBayesianAveraging/paper_examples
#python FullyBayesianBMADecomposition.py
#python StatisticalQuadraticModels1D.py
#python KernelDisplay.py
python FullyBayesianBMAO1D.py
