# NetInfoF

## Usage

1. Install required packages:
   
`pip install -r requirements.txt`

2. Compile random walk code:
   
`g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/rwcpp.cpp -o src/rwcpp$(python3-config --extension-suffix)`

3. Run code:

`bash run.sh`

## Details

`src/main_ours.py`: is used to train NetInfoF.

`src/main_batch.py`: is used to train NetInfoF with mini-batch on large graphs.