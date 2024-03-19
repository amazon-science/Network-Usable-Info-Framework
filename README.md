# NetInfoF Framework: Measuring and Exploiting Network Usable Information

Lee, M.C., Yu, H., Zhang, J., Ioannidis, V.N., Song, X., Adeshina, S., Zheng, D., and Faloutsos, C., “NetInfoF Framework: Measuring and Exploiting Network Usable Information”. International Conference on Learning Representations (ICLR), 2024. (Spotlight)

https://openreview.net/forum?id=KY8ZNcljVU

Please cite the paper as:

    @inproceedings{
      lee2024netinfof,
      title={NetInfoF Framework: Measuring and Exploiting Network Usable Information},
      author={Meng-Chieh Lee and Haiyang Yu and Jian Zhang and Vassilis N. Ioannidis and Xiang song and Soji Adeshina and Da Zheng and Christos Faloutsos},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=KY8ZNcljVU}
      }

## Introduction
to do

## Experiments
to do

## Usage

1. Install required packages:
   
`pip install -r requirements.txt`

2. Compile C++ random walk code:
   
`g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/rwcpp.cpp -o src/rwcpp$(python3-config --extension-suffix)`

3. Run code:

`bash run.sh`

## Details

`src/main_ours.py`: is used to train NetInfoF.

`src/main_batch.py`: is used to train NetInfoF with mini-batch on large graphs.
