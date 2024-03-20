"""
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
"""

from models.proposed import NetInfoF


def load_model(args, g, model_name, use_U=True, use_R=True, use_F=True, use_P=True, use_S=True):
    if model_name == 'NetInfoF':
        return NetInfoF(args, g, use_U, use_R, use_F, use_P, use_S)
    else:
        raise NotImplementedError
