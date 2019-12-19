from setuptools import setup, Extension
from torch.utils import cpp_extension

m_name = 'batch_find_neighbors'
setup(name=m_name,
      ext_modules=[
          cpp_extension.CppExtension(
              name='batch_find_neighbors', 
              sources=['batch_find_neighbors.cpp', 'neighbors/neighbors.cpp', 'cpp_utils/cloud/cloud.cpp'])
            ],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )
