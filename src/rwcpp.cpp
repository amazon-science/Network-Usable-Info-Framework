/* 
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <array>
#include <numeric>
#include <unordered_map>
#include <tuple>
#include <omp.h>

// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) rwcpp.cpp -o rwcpp$(python3-config --extension-suffix)


namespace std{
    namespace
    {

        template <class T>
        inline void hash_combine(std::size_t& seed, T const& v)
        {
            seed ^= hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }

        // Recursive template code derived from Matthieu M.
        template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct HashValueImpl
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
            hash_combine(seed, get<Index>(tuple));
          }
        };

        template <class Tuple>
        struct HashValueImpl<Tuple,0>
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            hash_combine(seed, get<0>(tuple));
          }
        };
    }

    template <typename ... TT>
    struct hash<std::tuple<TT...>>
    {
        size_t
        operator()(std::tuple<TT...> const& tt) const
        {
            size_t seed = 0;
            HashValueImpl<std::tuple<TT...> >::apply(seed, tt);
            return seed;
        }

    };
}

int add(int n, std::vector<int>& arr) {
    int res = 0;
    for (int i = 0; i < n; ++i) {
        res += arr[i];
    }
    return res; //n + n1;
}

/* Random walk from each node. */
std::unordered_map<std::tuple<int, int>, int> random_walks(const std::map<int, std::vector<int>>& neighbors,
                                            const int walk_length, const int num_trials, const int threshold){
  // contains random walk for each node across trials
  std::unordered_map<std::tuple<int, int>, int> output;

    #pragma omp parallel
    {
        size_t cnt = 0;
        for(auto node = neighbors.begin(); node !=neighbors.end(); ++node, cnt++)
        { 
            // do walk for each of this curr_node
            int curr_node = node->first;

            if (neighbors.at(curr_node).size() == 0) {
                continue;
            }

            for (int i=0; i < num_trials; i++){
                std::vector<float> walk;
                walk.push_back(curr_node);
                int last_visited_node = walk.back();
                std::vector<int> curr_neigh = neighbors.at(last_visited_node);
                // select random element from the neighbors
                int nex = curr_neigh[rand() % curr_neigh.size()];
                walk.push_back(nex);
                // update count in the global return map
                std::tuple<int, int> dict_key = std::make_tuple(curr_node, nex);
                output[dict_key]++ ;

                while(walk.size() < walk_length){
                    int walk_size = walk.size();
                    int cur = walk[walk_size - 1];
                    int prev = walk[walk_size - 2];
                    std::vector<int> curr_neigh = neighbors.at(cur);
                    std::vector<int> copy_curr_neigh;
                    copy_curr_neigh.assign(curr_neigh.begin(), curr_neigh.end());
                    copy_curr_neigh.erase(std::remove(copy_curr_neigh.begin(), copy_curr_neigh.end(), prev),
                                            copy_curr_neigh.end());
                    if (copy_curr_neigh.size() < 1) {
                        break;
                    }
                    // select random element from the neighbors
                    int nex = copy_curr_neigh[rand() % copy_curr_neigh.size()];
                    // update count in the global return map. curr_node is the walk start node here
                    std::tuple<int, int> dict_key = std::make_tuple(curr_node, nex);
                    output[dict_key]++ ;

                    // append nex to the walk
                    walk.push_back(nex);
                }

            }
        }
  }

  std::unordered_map<std::tuple<int, int>, int> new_output;
  for (auto const& [k, v] : output) {
	  if (v > threshold) {
		  new_output[k] = v;
	  }
  }

  return new_output;
}

namespace py = pybind11;

PYBIND11_MODULE(rwcpp, m) {
    m.def("add", &add, "A function which adds two numbers");
    m.def("random_walks", &random_walks, "Trial function");
}
