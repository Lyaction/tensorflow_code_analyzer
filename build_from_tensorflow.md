# macos 编译 tensorflow1.12 版本记录
## 准备工作
1、下载 tensorflow1.12 源码
  ``git clone https://github.com/tensorflow/tensorflow.git``
2、安装 python 环境
  ``
  conda create env=$name python=2.7
  conda install tensorflow=1.12
  ``
3、安装 bazel 0.14.1
  ``
  wget https://github.com/bazelbuild/bazel/releases/download/0.14.1/bazel-0.14.1-installer-darwin-x86_64.sh
  chmod +x bazel-0.14.1-installer-darwin-x86_64.sh
  sh bazel-0.14.1-installer-darwin-x86_64.sh
  ``
4、修改 configure.py 
  check_bazel_version('0.15.0') -> check_bazel_version('0.14.0')
5、修改 WORKSPACE
  check_bazel_version_at_least("0.15.0") -> check_bazel_version_at_least("0.14.0")

## 编译
``
  ./configure
  bazel build --verbose_failures --config=opt //tensorflow/tools/pip_package:build_pip_package
``
