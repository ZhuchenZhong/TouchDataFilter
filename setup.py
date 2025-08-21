from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sysconfig

# 获取Python库路径
python_lib = sysconfig.get_config_var('LIBDIR')
python_version = sysconfig.get_config_var('py_version_short')
lib_name = f"python{python_version}"

ext_modules = [
    Extension(
        "predict_wrapper",
        ["predict_wrapper.pyx"],
        include_dirs=[np.get_include()],
        libraries=[lib_name],  # 使用动态库
        library_dirs=[python_lib],
        runtime_library_dirs=[python_lib],  # 添加运行时库路径
        extra_link_args=["-Wl,-rpath," + python_lib]  # 确保rpath正确
    )
]

setup(
    name="TouchDataFilter_wrapper",
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
)